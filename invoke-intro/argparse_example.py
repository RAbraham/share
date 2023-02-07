import argparse
from pathlib import Path
from activity import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Model")
    # TODO: choice
    parser.add_argument("-a", "--activity", required=True, choices=['train', 'predict'], help="train or predict")
    # TODO: default and type
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    # TODO: Boolean options 1
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, default=False, help="print debug "
                                                                                                    "statements")  # Python 3.9+
    # TODO: Boolean options 2
    parser.add_argument('-p', '--profile', action=argparse.BooleanOptionalAction, default=True, help="profile memory "
                                                                                                     "statements")  # Python 3.9+
    # TODO: Multiple Values
    parser.add_argument('-l', '--learning_rate', nargs="*", help="model learning rate")

    # TODO: Can't give -p here as it'll conflict with `-p` for profile
    # TODO: type Path
    parser.add_argument("-k", "--path", type=Path, help="output folder")
    args = parser.parse_args()

    if args.activity == 'train':
        train(epochs=args.epochs, debug=args.debug, profile=args.profile, learning_rates=args.learning_rate,
                       path=args.path)

