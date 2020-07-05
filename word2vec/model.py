
from nltk.corpus import stopwords
import os
import numpy as np
import sys
import re
from gensim.models import Word2Vec
import nltk
import pickle
import time
# nltk.download('stopwords')


class WordEmbeddingModel():

    def __init__(self, args):
        '''
            A wrapper class based on Gensim Word2Vec model.
            Required Libraries: re, gensim, nltk
        '''
        self.built = False
        self.args = args
        self.trained = False
        self.data = False
        self.sw = set(list(stopwords.words('english')) +
                      re.split('', r"!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"))
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
                             epochs=5)
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
        for text in (os.listdir(path)):
            with open(os.path.join(path, text), 'r') as file:
                texts.append(
                    self.preprocess(str(file.read()))
                )
                labels.append(text.split('.')[0])
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
            self.__train(self.preprocess(sentence), self.args['epochs'])
            embedding = self.model.wv[self.preprocess(sentence)]
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
        embeddings = list(map(self.embedd, [' '.join(x) for x in self.texts]))
        print(f'saving embeddings to {path}')
        with open(path, 'wb') as file:
            pickle.dump(embeddings, file)
        return embeddings

    def preprocess(self, text):
        text = text.lower()
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        word_token = nltk.word_tokenize(text)
        word_token = [word for word in word_token if word not in self.sw]
        return word_token
