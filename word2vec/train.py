import argparse
import pickle
import time
from model import WordEmbeddingModel

if __name__ == '__main__':
    time = time.time()
    parser = argparse.ArgumentParser('Text Embedding Training')
    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--save_model', type=str, default='.')
    parser.add_argument('--save_embedding', type=str)
    parser.add_argument('--size', type=int, default=200)
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('-min_count', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--sg', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)

    args = vars(parser.parse_args())

    with open('config'+str(time)+'.p', 'wb') as file:
        pickle.dump(args, file)
    model = WordEmbeddingModel(args)
    model.load_data(path=args['data_path'])
    model.build()
    model.save_model(args['save_model'])
    model.save_embeddings(args['save_embedding'])
