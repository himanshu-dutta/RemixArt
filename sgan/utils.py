import os
import time
import torch
import random
import numpy as np
import pickle as p
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

############################
# General Utility
############################


def load(path):
    with open(path, 'rb') as file:
        return p.load(file)


def save(obj, path):
    with open(path, 'wb') as file:
        p.dump(obj, file)


def print_styled(text):
    hf = '\n'+'#' * len(text) + '#' * 5 + '\n'
    sent = '#   ' + text
    print(hf + sent + hf)


############################
# Loss Functions
############################


def KL_Div(mu, var, fact):
    KLD = mu.pow(2).add_(var.exp()).mul_(-1).add_(1).add_(var)
    KLD = torch.mean(KLD).mul_(-0.5)
    return KLD*fact

############################
# Model Utility
############################


def adjust_lr(optimizer, epoch, init_lr, args):
    lr = init_lr * (0.1 ** (epoch // args['LR_DECAY_EPOCH']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_model(args, stage=1, path=None, device=None):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G_loss = None
    D_loss = None
    epoch = 0
    gen_para = []
    if stage == 1:
        gen = STAGE1_G(args).to(device)
        dis = STAGE1_D(args).to(device)
        if path:
            checkpoint = torch.load(path['stage1'])
            gen.load_state_dict(checkpoint['gen_state_dict'])
            dis.load_state_dict(checkpoint['dis_state_dict'])
        for p in gen.parameters():
            if p.requires_grad:
                gen_para.append(p)
        gen_optim = optim.Adam(
            gen_para, lr=args['GENERATOR_LR'], betas=(0.5, 0.999))
        dis_optim = optim.Adam(
            dis.parameters(), lr=args['DISCRIMINATOR_LR'], betas=(0.5, 0.999))
        if path:
            gen_optim.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            dis_optim.load_state_dict(checkpoint['dis_optimizer_state_dict'])
            G_loss = checkpoint['G_loss']
            D_loss = checkpoint['D_loss']
            epoch = checkpoint['epoch']

    elif stage == 2:
        gen1 = STAGE1_G(args).to(device)
        if path:
            checkpoint = torch.load(path['stage1'])
            gen1.load_state_dict(torch.load(path['gen1']))
        gen = STAGE2_G(gen1, args).to(device)
        dis = STAGE2_D(args).to(device)
        if path['stage2'] != '':
            checkpoint = torch.load(path['stage2'])
            gen.load_state_dict(checkpoint['gen_state_dict'])
            dis.load_state_dict(checkpoint['dis_state_dict'])
        for p in gen.parameters():
            if p.requires_grad:
                gen_para.append(p)
        gen_optim = optim.Adam(
            gen_para, lr=args['GENERATOR_LR'], betas=(0.5, 0.999))
        dis_optim = optim.Adam(
            dis.parameters(), lr=args['DISCRIMINATOR_LR'], betas=(0.5, 0.999))
        if path['stage2'] != '':
            gen_optim.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            dis_optim.load_state_dict(checkpoint['dis_optimizer_state_dict'])
            G_loss = checkpoint['G_loss']
            D_loss = checkpoint['D_loss']
            epoch = checkpoint['epoch']

    if not path:
        gen.apply(weights_init)
        dis.apply(weights_init)

    return gen, dis, gen_optim, dis_optim, G_loss, D_loss, epoch


def save_model(gen, dis, gen_optim, dis_optim, G_loss, D_loss, args, epochs, timestamp=None):
    checkpoint = {
        'epoch': epochs,
        'gen_state_dict': gen.state_dict(),
        'dis_state_dict': dis.state_dict(),
        'gen_optimizer_state_dict': gen_optim.state_dict(),
        'dis_optimizer_state_dict': dis_optim.state_dict(),
        'G_loss': G_loss,
        'D_loss': D_loss,
    }

    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
    dir_ = args['SAVE_MODEL']
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    torch.save(
        checkpoint,
        os.path.join(dir_, f"model_{args['STAGE']}_{epochs}.tar"))
    print_styled(
        f"Model for stage {args['STAGE']} for epoch {epochs} saved at {dir_}...")
    return os.path.join(dir_, f"model_{args['STAGE']}_{epochs}.tar")


def train(dataloader, args, path=None, device=None, timestamp=None, KL_factor=2):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # initializing the logger
    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
    writer_path = os.path.join(
        args['LOG_DIR'], timestamp, f"stage{args['STAGE']}")

    writer = SummaryWriter(writer_path)

    # loading the models
    epochs = args['MAX_EPOCH']
    stage = args['STAGE']
    gen, dis, gen_optim, dis_optim, G_loss, D_loss, last_epoch = load_model(
        args, stage, path, device)

    start = time.time()

    for epoch in tqdm(range(last_epoch, epochs)):
        print_styled(f'Running epoch:{epoch}...')

        G_loss_run = 0.0
        D_loss_run = 0.0

        adjust_lr(dis_optim, epoch, args['DISCRIMINATOR_LR'], args)
        adjust_lr(gen_optim, epoch, args['GENERATOR_LR'], args)

        for i, data in enumerate(dataloader):
            if i % 50 == 0:
                print(f'Currently running batch {i}')
            # loading each batch
            text, audio, image = data
            text, audio, image = text.to(device), audio.to(device),\
                image.to(device)
            BATCHSIZE = text.shape[0]

            noise = torch.randn(BATCHSIZE, args['Z_DIM']).to(device)

            # generating labels for data
            zeros = torch.zeros(int(BATCHSIZE), 1).to(device)
            ones = torch.ones(int(BATCHSIZE), 1).to(device)

            # # logging the model
            # if epoch == 0 and i == 0:
            #     writer.add_graph(gen, (text, audio, noise))

            # generating discriminator data
            _, img_f, mu_f, logvar_f = gen(text, audio, noise)
            D_real = dis(image.detach())
            D_fake = dis(img_f.detach())

            # calculating discriminator loss
            D_real_loss = F.binary_cross_entropy(D_real, ones.view(-1, 1))
            D_fake_loss = F.binary_cross_entropy(D_fake, zeros.view(-1, 1))

            D_loss = D_real_loss + D_fake_loss

            # backprop for discriminator network
            dis_optim.zero_grad()
            D_loss.backward()
            dis_optim.step()

            # feed-forward for generator
            D_fake = dis(img_f)

            # calculating generator loss
            KL_loss = KL_Div(mu_f, logvar_f, KL_factor)
            G_loss_fake = F.binary_cross_entropy(D_fake, ones)
            G_loss = G_loss_fake + KL_loss

            # backprop for generator network
            gen_optim.zero_grad()
            G_loss.backward()
            gen_optim.step()

            G_loss_run += G_loss.item()
            D_loss_run += D_loss.item()
            if i % args['DATAFOLDS'] == 0:
                writer.add_scalar('D_loss', D_loss.flatten()
                                  [0], epoch*BATCHSIZE+i)
                writer.add_scalar('D_real_loss', D_real_loss.flatten()[
                                  0], epoch*BATCHSIZE+i)
                writer.add_scalar('D_fake_loss', D_fake_loss.flatten()[
                                  0], epoch*BATCHSIZE+i)
                writer.add_scalar('G_loss', G_loss.flatten()
                                  [0], epoch*BATCHSIZE+i)
                writer.add_scalar('G_loss_fake', G_loss_fake.flatten()[
                                  0], epoch*BATCHSIZE+i)
                writer.add_scalar('KL_loss', KL_loss.flatten()[
                                  0], epoch*BATCHSIZE+i)

        if epoch % args['SNAPSHOT_INTERVAL'] == 0:
            saved_model = save_model(
                gen, dis, gen_optim, dis_optim, G_loss, D_loss, args, epoch, timestamp)
            #########################
            #  specific to cloud env
            #########################
            # upload_blob(saved_model, saved_model)
            # upload_local_directory_to_gcs('logs', 'logs')
            # os.remove(saved_model)

        # printing loss after each epoch
        print_styled('Epoch:{},   G_loss:{},   D_loss:{}'.format(
            epoch, G_loss_run/(i+1), D_loss_run/(i+1)))
        plt.imshow(np.moveaxis(img_f.cpu()[0].detach().numpy(), 0, 2))
        plt.show()
    save_model(gen, dis, args, epoch, timestamp)
    print_styled(f'Model training took {(time.time()-start)/60} mins.')
    writer.close()
    return gen, dis


def predict(text, audio, args, path, stage, device=None):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen, _ = load_model(args, stage, path, device)
    noise = torch.randn(text.shape[0], args['Z_DIM']).to(device)
    res = gen.eval(text, audio, noise).detach().numpy()
    if len(res.shape) == 4:
        return np.moveaxis(res, 1, 3)
    elif len(res.shape) == 3:
        return np.moveaxis(res, 0, 2)


############################
# Data Utility
############################


class DataSet(Dataset):

    def __init__(self, args):
        self.folds = args['DATAFOLDS']
        if args['STAGE'] == 1:
            self.imgdim = args['IMGSIZE1']
        else:
            self.imgdim = args['IMGSIZE2']
        self.transform = transforms.Compose([
            transforms.Resize(self.imgdim),
            transforms.ToTensor(),
        ])
        self.__load_embeddings(args['DATA_DIR'])
        self.__load_images(args['IMG_DIR'])

    def __len__(self):
        return len(self.text_embedding) * self.folds

    def __load_embeddings(self, path):
        self.text_embedding, self.text_labels = load(
            os.path.join(path, 'text_embeddings.p'))
        self.audio_embedding, self.audio_labels = load(
            os.path.join(path, 'audio_embeddings.p'))
        self.mapping = pd.read_csv(os.path.join(path, 'mapping.csv'))
        self.mapping['id'] = self.mapping['id'].apply(str)  # .set_index('id')

    def __load_images(self, path):
        classes = os.listdir(path)
        self.label_map = {cls.lower(): i for i, cls in enumerate(classes)}
        classes = [os.path.join(path, cls) for cls in classes]
        self.images = ([[os.path.join(cls, file) for file in os.listdir(
            cls) if file.endswith('.jpg') or file.endswith('.png')] for cls in classes])

    def __process_image(self, path):
        img = Image.open(path)
        return self.transform(img)

    def __getitem__(self, idx):
        '''
            for every embedding it returns 50% mismatched data
            and 50% matching data
            depends on the tweak with neg
        '''
        neg = False
        if idx % self.folds / self.folds >= 0.5:
            neg = False
        idx = idx // self.folds

        text = torch.tensor(self.text_embedding[idx])
        audio = torch.tensor(self.audio_embedding[idx])

        id = self.text_labels[idx]
        genre = list(self.mapping[self.mapping['id'] == id].genre)[0].lower()
        genre_idx = self.label_map[genre]

        if neg:
            cls = random.choice(
                [i for i in range(len(self.label_map)) if i != genre_idx])
            img_idx = random.choice(range(len(self.images[cls])))
        else:
            cls = genre_idx
            img_idx = random.choice(range(len(self.images[cls])))

        img = self.__process_image(self.images[cls][img_idx])

        return text.type(torch.float32), audio.type(torch.float32), img.type(torch.float32)
