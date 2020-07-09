import os
import time
import torch
import random
import numpy as np
import pickle as p
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from model import STAGE1_G, STAGE1_D, STAGE2_G, STAGE2_D, weights_init

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
    hf = '#' * len(text) + '#' * 5 + '\n'
    sent = '#   ' + text + '\n'
    print(hf + sent + hf)


############################
# Loss Functions
############################


def KL_Div(mu, var):
    KLD = mu.pow(2).add_(var.exp()).mul_(-1).add_(1).add_(var)
    KLD = torch.mean(KLD).mul_(-0.5)
    return KLD


def G_Loss(g):
    pass


def D_Loss(d):
    pass


############################
# Model Utility
############################


def load_model(args, stage=1, path=None, device=None):
    if not device:
        device = torch.device('cpu')

    if stage == 1:
        gen = STAGE1_G(args).to(device)
        dis = STAGE1_D(args).to(device)
        if path:
            gen.load_state_dict(torch.load(path['gen1']))
            dis.load_state_dict(torch.load(path['dis1']))

    elif stage == 2:
        gen1 = STAGE1_G(args)
        if path:
            gen1.load_state_dict(torch.load(path['gen1']))
        gen = STAGE2_G(gen1, args)
        dis = STAGE2_D(args)
        if path['gen2'] != '':
            gen.load_state_dict(torch.load(path['gen2']))
            dis.load_state_dict(torch.load(path['dis2']))

    gen.apply(weights_init)
    dis.apply(weights_init)

    return gen, dis


def train(dataloader, args, path=None, device=None, summary=None):
    if not device:
        device = torch.device('cpu')

    # loading the models
    epochs = args['MAX_EPOCH']
    stage = args['STAGE']
    gen, dis = load_model(args, stage, path, device)

    start = time.time()

    # optimizer and loss
    gen_optim = optim.Adam(
        gen.parameters(), lr=args['GENERATOR_LR'], betas=(0.5, 0.999))
    dis_optim = optim.Adam(
        dis.parameters(), lr=args['DISCRIMINATOR_LR'], betas=(0.5, 0.999))

    for epoch in tqdm(range(epochs)):
        print_styled(f'Running epoch:{epoch}...')

        G_loss_run = 0.0
        D_loss_run = 0.0

        for i, data in enumerate(dataloader):

            # loading each batch
            text, audio, image, label = data
            text, audio, image, label = text.to(device), audio.to(device),\
                image.to(device), label.to(device)
            BATCHSIZE = text.shape[0]

            # segregating mismatched and real
            mis = (label == 0).nonzero().flatten()
            real = (label != 0).nonzero().flatten()

            image_r, label_r = image[real], label[real]
            text_m, audio_m, image_m, label_m = text[mis], audio[mis], image[mis], label[mis]
            noise_m = torch.randn(len(label_m), args['Z_DIM']).to(device)

            # generating labels for fake data
            label_f = torch.zeros(int(BATCHSIZE*0.5), 1).to(device)
            label_t = torch.ones(int(BATCHSIZE*0.5), 1).to(device)

            # generating random noise from stdnormal dist
            noise = torch.randn(int(BATCHSIZE*0.5), args['Z_DIM']).to(device)

            # generating discriminator data
            D_real = dis(image_r)
            D_mis = dis(gen(text_m, audio_m, noise_m))
            D_fake = dis(gen(text[:int(BATCHSIZE*0.5)],
                             audio[int(BATCHSIZE*0.5):], noise))

            # calculating discriminator loss
            D_real_loss = F.binary_cross_entropy(D_real, label_r.view(-1, 1))
            D_mis_loss = F.binary_cross_entroy(D_mis, label_m.view(-1, 1))
            D_fake_loss = F.binary_cross_entropy(D_fake, label_f.view(-1, 1))

            D_loss = D_real_loss + D_mis_loss + D_fake_loss

            # backprop for discriminator network
            dis_optim.zero_grad()
            D_loss.backward()
            dis_optim.step()

            # feed-forward for generator

            noise = torch.randn(int(BATCHSIZE*1.5), args['Z_DIM']).to(device)

            D_fake = dis(gen(text[:int(BATCHSIZE*0.5)],
                             audio[int(BATCHSIZE*0.5):], noise))

            # calculating generator loss
            G_loss = F.binary_cross_entropy(D_fake, label_t)

            # backprop for generator network
            g_optim.zero_grad()
            G_loss.backward()
            g_optim.step()

            G_loss_run += G_loss.item()
            D_loss_run += D_loss.item()

        # printing loss after each epoch
        print('Epoch:{},   G_loss:{},   D_loss:{}'.format(
            epoch, G_loss_run/(i+1), D_loss_run/(i+1)))


def predict():
    pass


def save_model():
    pass


############################
# Data Utility
############################


class DataSet(Dataset):

    def __init__(self):
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
        self.text_embedding, self.audio_labels = load(
            os.path.join(path, 'text_embeddings.p'))
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
        '''
        neg = False
        if idx % self.folds / self.folds >= 0.5:
            neg = True
        idx = idx // self.folds

        text = torch.tensor(self.text_embedding[idx])
        audio = torch.tensor(self.text_embedding[idx])

        id = self.text_labels[idx]
        genre = list(self.mapping[self.mapping['id'] == id].genre)[0].lower()
        genre_idx = self.label_map[genre]

        if neg:
            cls = random.choice(
                [i for i in range(len(self.label_map)) if i != genre_idx])
            img_idx = random.choice(range(len(self.images[cls])))
            label = torch.tensor(0)
        else:
            cls = genre_idx
            img_idx = random.choice(range(len(self.images[cls])))
            label = torch.tensor(1)

        img = self.__process_image(self.images[cls][img_idx])

        return text, audio, img, label
