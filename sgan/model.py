import torch
import torch.nn as nn
import torch.nn.functional as F


############################
# Weight Initialization
# for initializing weights
# from normal dist
############################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

########################################################
# Helper Models:
# 1. T2O converts 2D embedding representation to 1D
# 2. CA_NET is the model used for Conditionally
# Augmenting both text and audio embedding together
########################################################


class T2O(nn.Module):
    def __init__(self, args):
        '''
            The word embeddings from the Word2Vec Model are of a 
            constant dimension, 100x200. This class is to convert 
            the 2D representation to 1D, so as to pass it through 
            rest of the model.
        '''

        super(T2O, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # number of channels for 1D conv.
        self.in_channels = args['EMBEDDDIM'][0]
        # Calculating the inputs for the linear layer.
        self.in_features = (args['EMBEDDDIM'][1]-10+1)
        self.model = nn.Sequential(
            nn.Conv1d(self.in_channels, 32, 10),
            nn.Flatten(),
            nn.Linear(32*self.in_features, 1024),
            nn.ReLU()
        ).to(self.device)

    def forward(self, embedding):
        print(embedding.device)
        return self.model(embedding)


class CA_NET(nn.Module):
    def __init__(self, args):
        super(CA_NET, self).__init__()
        self.t_dim = args['DIMENSION']
        self.c_dim = args['CONDITION_DIM']
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding, audio_embedding):
        x = self.relu(self.fc(text_embedding))
        y = self.relu(self.fc(audio_embedding))
        mu = torch.cat((x[:, :self.c_dim], y[:, :self.c_dim]), 1)
        logvar = torch.cat((x[:, self.c_dim:], y[:, self.c_dim:]), 1)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        print(mu.device, logvar.device)
        std = logvar.mul(0.5).exp_()
        eps = torch.Tensor(std.size()).to(self.device).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding, audio_embedding):
        mu, logvar = self.encode(text_embedding, audio_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


########################################################
# STAGE 1:
########################################################

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Upsale the spatial size by a factor of 2


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True)
    )
    return block


class STAGE1_G(nn.Module):
    def __init__(self, args):
        super(STAGE1_G, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.gf_dim = args['GF_DIM'] * 8
        self.ef_dim = args['CONDITION_DIM'] * 2
        self.z_dim = args['Z_DIM']
        self.args = args
        self.define_module()

    def define_module(self):
        # twice, once for audio vector and once for the text
        ninput = self.z_dim + self.ef_dim
        ngf = self.gf_dim
        # extracts a vector of 1D from 2D
        self.t2o = T2O(self.args).to(self.device)
        # conditional aug network
        self.ca_net = CA_NET(self.args).to(self.device)
        # ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True),
        ).to(self.device)
        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())

    def forward(self, text_embedding, audio_embedding, noise):
        print('Here1')
        self.fc.train()
        # 2d to 1d vector
        print('Here2')
        t_e = self.t2o(text_embedding)
        print('Here3')
        a_e = self.t2o(audio_embedding)
        # conditional augmentation
        print('Here4')
        c_code, mu, logvar = self.ca_net(t_e, a_e)
        z_c_code = torch.cat((noise, c_code), 1)
        # linear and upsampling layers
        h_code = self.fc(z_c_code)
        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        # image generation
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar

    def eval(self, text_embedding, audio_embedding, noise):
        self.fc.eval()
        # 2d to 1d vector
        t_e = self.t2o(text_embedding)
        a_e = self.t2o(audio_embedding)
        # conditional augmentation
        c_code, mu, logvar = self.ca_net(t_e, a_e)
        z_c_code = torch.cat((noise, c_code), 1)
        # linear and upsampling layers
        h_code = self.fc(z_c_code)
        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        # image generation
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar


class STAGE1_D(nn.Module):
    def __init__(self, args):
        super(STAGE1_D, self).__init__()
        self.df_dim = args['DF_DIM']
        self.ef_dim = args['CONDITION_DIM']
        self.args = args
        self.define_module()

    def define_module(self):

        ndf = self.df_dim

        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.outlogits = nn.Sequential(
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Sigmoid())
        # self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        # self.get_uncond_logits = None

    def forward(self, image):
        self.encode_img.train()
        self.outlogits.train()
        img_embedding = self.encode_img(image)
        img = self.outlogits(img_embedding)
        return img

    def eval(self, image):
        self.encode_img.eval()
        self.outlogits.eval()
        img_embedding = self.encode_img(image)
        img = self.outlogits(img_embedding)
        return img

########################################################
# STAGE 2:
########################################################


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G, args):
        super(STAGE2_G, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.gf_dim = args['GF_DIM']
        self.ef_dim = args['CONDITION_DIM']*2
        self.z_dim = args['Z_DIM']
        self.args = args
        self.STAGE1_G = STAGE1_G
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.args['R_NUM']):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # extracts a vector of 1D from 2D
        self.t2o = T2O(self.args).to(self.device)
        # conditional aug network
        self.ca_net = CA_NET(self.args).to(self.device)
        # --> 4ngf x 16 x 16
        self.encoder = nn.Sequential(
            conv3x3(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)).to(self.device)
        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)).to(self.device)
        self.residual = self._make_layer(ResBlock, ngf * 4).to(self.device)
        # --> 2ngf x 32 x 32
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock(ngf * 2, ngf)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock(ngf, ngf // 2)
        # --> ngf // 4 x 256 x 256
        self.upsample4 = upBlock(ngf // 2, ngf // 4)
        # --> 3 x 256 x 256
        self.img = nn.Sequential(
            conv3x3(ngf // 4, 3),
            nn.Tanh())

    def forward(self, text_embedding, audio_embedding, noise):
        # setting all the sequential models to training
        self.encoder.train()
        self.hr_joint.train()
        self.residual.train()
        _, stage1_img, _, _ = self.STAGE1_G(
            text_embedding, audio_embedding, noise)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)

        # 2d to 1d vector
        t_e = self.t2o(text_embedding)
        a_e = self.t2o(audio_embedding)
        # conditional augmentation
        c_code, mu, logvar = self.ca_net(t_e, a_e)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        fake_img = self.img(h_code)
        return stage1_img, fake_img, mu, logvar

    def eval(self, text_embedding, audio_embedding, noise):
        # setting all the sequential models to evaluation
        self.encoder.eval()
        self.hr_joint.eval()
        self.residual.eval()
        _, stage1_img, _, _ = self.STAGE1_G.eval(
            text_embedding, audio_embedding, noise)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)

        # 2d to 1d vector
        t_e = self.t2o(text_embedding)
        a_e = self.t2o(audio_embedding)
        # conditional augmentation
        c_code, mu, logvar = self.ca_net(t_e, a_e)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        fake_img = self.img(h_code)
        return stage1_img, fake_img, mu, logvar


class STAGE2_D(nn.Module):
    def __init__(self, args):
        super(STAGE2_D, self).__init__()
        self.df_dim = args['DF_DIM']
        self.ef_dim = args['CONDITION_DIM'] * 2
        self.args = args
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  # 128 * 128 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 64 * 64 * ndf * 2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
            conv3x3(ndf * 32, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
            conv3x3(ndf * 16, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 8
        )
        self.outlogits = nn.Sequential(
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Sigmoid())

        # self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
        # self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, image):
        self.encode_img.train()
        self.outlogits.train()

        img_embedding = self.encode_img(image)
        img = self.outlogits(img_embedding)
        return img

    def eval(self, image):
        self.encode_img.eval()
        self.outlogits.eval()

        img_embedding = self.encode_img(image)
        img = self.outlogits(img_embedding)
        return img
