from parse_args import parse_args
import os
from train_network import train
from inference import predict

def run():
    """Runs dilated residual network model in either train or predict mode"""

    config = parse_args()

    if not os.path.isdir(config.logs):
        os.makedirs(config.logs)

    if config.mode == 'train':
        train(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        ValueError("Mode must be either train or predict")


if __name__ == '__main__':
    run()