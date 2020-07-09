import os
import random
import torch
import numpy as np
import pickle as p
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

############################
# General Utility
############################


def load(path):
    with open(path, 'rb') as file:
        return p.load(file)


def save(obj, path):
    with open(path, 'wb') as file:
        p.dump(obj, file)

############################
# Model Utility
############################


def train():
    pass


def predict():
    pass


def save_model():
    pass


def load_model():
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
            self.imdim = args['IMGSIZE2']
        self.transform = transforms.Compose([
            transforms.Resize(self.imgdim),
            transforms.ToTensor(),
        ])
        self.__load_embeddings(args['DATA_DIR'])
        self.__load_images(args['IMG_DIR'])
        print(len(self.text_embedding) * self.folds)

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
            for every embedding it returns 40% mismatched data
            and 60% matching data
        '''
        neg = False
        if idx % self.folds / self.folds >= 0.6:
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
            label = 0
        else:
            cls = genre_idx
            img_idx = random.choice(range(len(self.images[cls])))
            label = 1

        img = self.__process_image(self.images[cls][img_idx])

        return text, audio, img, label
