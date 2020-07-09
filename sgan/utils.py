import os
import numpy as np
import pickle as p
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
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


class Training():
    pass


def predict():
    pass


############################
# Data Utility
############################

class DataSet(DataLoader):

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

    def __load_embeddings(self, path):
        self.text_embedding, self.text_labels = load(
            os.path.join(path, 'text_embeddings.p'))
        self.text_embedding, self.audio_labels = load(
            os.path.join(path, 'text_embeddings.p'))
        self.mapping = pd.read_csv(os.path.join(path, 'mapping.csv'))

    def __load_images(self, path):
        classes = os.listdir(path)
        self.label_map = {cls.lower(): i for i, cls in enumerate(classes)}
        classes = [os.path.join(path, cls) for cls in classes]
        self.images = ([[os.path.join(cls, file) for file in os.listdir(
            cls) if file.endswith('.jpg') or file.endswith('.png')] for cls in classes])

    def __process_image(self, path):
        img = Image.open(path)
        return self.transform(img)

    # def __getitem__(self, index):
