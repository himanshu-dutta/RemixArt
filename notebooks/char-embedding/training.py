import argparse
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Character Embedding Training')
    parser.add_argument('--vocab',
                        type=str,
                        default="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}\n")
    parser.add_argument('-max_length', type=int, default=1024)
    parser.add_argument('--embedding_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)

    args = parser.parse_args()

    with open('config.p', 'wb') as file:
        pickle.dump(args, file)
