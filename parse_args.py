import argparse

class Config:
    pass

def parse_args():
    """Parses input arguments into a configuration class

    Args:
        args: Argument parser
    Returns:
        config: Instance of configuration class
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help="Mode must be either train or predict")
    parser.add_argument('--network', type=str, default='DRN18', help="Select either DRN18 or DRN26 for network")
    parser.add_argument('--image_dims', type=int, default=500, help="Dimension of the input image, assumes square")
    parser.add_argument('--channels', type=int, default=3, help="Dimension of the image color channel")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--directory', type=str, help="Enter the path to the directory contanining all images")
    parser.add_argument('--train_file', type=str, help="Path to the training data file")
    parser.add_argument('--val_file', type=str, help="Path to the validation data file")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes to predict")
    parser.add_argument('--loss', type=str, default='dice', help="Enter the loss function to use; either 'dice' or weighted cross-entropy - 'CE'")
    parser.add_argument('--logs', type=str, default='logs', help="Enter the path to the log/save directory")
    parser.add_argument('--use_weights', type=bool, default=False, help="Flag indicating whether to include a weight map")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate during training")
    args = parser.parse_args()

    config = Config()
    config.mode = args.mode
    config.network = args.network
    config.directory = args.directory
    config.loss = args.loss
    config.num_epochs = args.num_epochs
    config.image_dims = [args.image_dims, args.image_dims, args.channels]
    config.batch_size = args.batch_size
    config.num_classes = args.num_classes
    config.use_weights = args.use_weights
    config.train_file = args.train_file
    config.val_file = args.val_file
    config.logs = args.logs
    config.learning_rate = args.learning_rate

    return config