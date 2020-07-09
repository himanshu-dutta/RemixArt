import argparse
import sys
sys.path.append('.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SGAN Model Training')

    # data parameters
    parser.add_argument('--DATA_DIR', type=str)
    parser.add_argument('--IMG_DIR', type=str)
    parser.add_argument('--STAGE1_G', type=str)
    parser.add_argument('--DATAFOLDS', type=int)

    # run parameters
    parser.add_argument('--CUDA', type=int, default=0)
    parser.add_argument('--WORKERS', type=int, default=4)
    parser.add_argument('--BATCH_SIZE', type=int, default=128)
    parser.add_argument('--MAX_EPOCH', type=int, default=5)
    parser.add_argument('--STAGE', type=int, default=1)
    parser.add_argument('--SNAPSHOT_INTERVAL', type=int, default=2)
    parser.add_argument('--LR_DECAY_EPOCH', type=int, default=20)

    # image specifications
    parser.add_argument('--IMGSIZE1', type=int, default=64)
    parser.add_argument('--IMGSIZE2', type=int, default=256)

    # model architecture definitions
    parser.add_argument('--EMBEDDDIM', type=tuple, default=(100, 200))
    parser.add_argument('--DIMENSION', type=int, default=1024)
    parser.add_argument('--CONDITION_DIM', type=int, default=128)
    parser.add_argument('--R_NUM', type=int, default=4)
    parser.add_argument('--DF_DIM', type=int, default=96)
    parser.add_argument('--GF_DIM', type=int, default=192)
    parser.add_argument('--Z_DIM', type=int, default=100)

    # model hyperparameters
    parser.add_argument('--DISCRIMINATOR_LR', type=float, default=0.0002)
    parser.add_argument('--GENERATOR_LR', type=float, default=0.0002)

    args = parser.parse_args()
    args = vars(args)
    print(args)
