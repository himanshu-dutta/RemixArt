import time
import pickle
from gensim.models import Word2Vec
import re
import sys
import numpy as np
import os
from nltk.corpus import stopwords
import nltk
import ast
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')


class AudioEmbeddingModel():

    def __init__(self, args):
        '''
            A wrapper class based on Gensim Word2Vec model.
            Required Libraries: re, gensim, nltk
        '''
        self.built = False
        self.args = args
        self.trained = False
        self.data = False
        print('embedding model initialized with parameters:\n', args)

    def build(self):
        '''
            Builds a new model with parameters specified in args.
        '''
        if not self.built and self.data:
            self.model = Word2Vec(
                # self.texts,
                size=self.args['size'],
                min_count=self.args['min_count'],
                workers=self.args['workers'],
                sg=self.args['sg'],
            )
            self.model.build_vocab(self.texts)
            self.model.train(self.texts,
                             total_examples=len(self.texts),
                             epochs=self.args['epochs'])
            self.built = True
        else:
            print('model alreay built')

    def load_model(self, path):
        '''
            Loads the existing model from specified path
        '''
        if self.build:
            print('A model exists. Still want to load?(y or n')
            if str(input()) == 'y':
                self.model = Word2Vec.load(path)
                self.built = True
        else:
            self.model = Word2Vec.load(path)
            self.built = True

    def save_model(self, path):
        self.model.save(os.path.join(path, f'model_{time.time()}.model'))

    def load_data(self, path):
        texts = []
        labels = []
        print(f'loading data from {path}...')
        df = pd.read_csv(path)
        texts = df['notes'].to_list()
        texts = [ast.literal_eval(str(text)) for text in texts]
        labels = df['id'].to_list()
        self.texts = texts
        self.labels = labels
        self.data = True
        print(f'Total {len(texts)} texts loaded...')

    def __train(self, text, epochs=5):
        if self.data:
            self.text = text
            self.model.train(
                self.text,
                total_examples=len(self.texts),
                epochs=epochs
            )
        else:
            print('Load the data first.')

    def embedd(self, sentence):
        if self.build:
            self.__train(sentence)
            embedding = self.model.wv[sentence]
            if len(embedding) > self.args['embedding_size']:
                embedding = embedding[:self.args['embedding_size']]
            elif 0 < len(embedding) < self.args['embedding_size']:
                embedding = np.concatenate(
                    (embedding, np.zeros((self.args['embedding_size'] - len(embedding), self.args['size']), dtype=np.float64)))
            elif len(embedding) == 0:
                embedding = np.zeros(
                    (self.args['embedding_size'], self.args['size']), dtype=np.float64)
            return embedding
        else:
            print("Build the model first.")

    def save_embeddings(self, path):
        embeddings = list(map(self.embedd, self.texts))
        print(f'saving embeddings to {path}')
        with open(path, 'wb') as file:
            pickle.dump((embeddings, self.labels), file)
        return embeddings
