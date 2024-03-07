import argparse
import json


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir',
                        help='Path to trained model',
                        required=True,
                        metavar='DIR')

    parser.add_argument('--control_file',
                        help='control file for apply mode',
                        metavar='FOO.json',
                        required=True)

    parser.add_argument('--convert_fractions',
                        help='convert 1/4 1/2 3/4 to single unicode characters',
                        action='store_true',
                        default=False)

    parser.add_argument('--output_dir',
                        help='output file for apply mode',
                        metavar='DIR',
                        required=True)

    return parser.parse_args()


