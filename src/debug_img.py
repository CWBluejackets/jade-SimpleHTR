import argparse

from path import Path

import locations
from dataloader_iam import DataLoaderIAM
from main import get_img_size
from preprocessor import Preprocessor


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        help='Directory containing IAM dataset.',
                        type=Path,
                        required=True)
    parser.add_argument('--debug_dir',
                        help='directory to dump CV2',
                        type=Path,
                        required=True)
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)

    return parser.parse_args()


def main_fn():
    # parse arguments and set model location and CTC decoder
    args = parse_args()
    locations.set_debug_dir(args.debug_dir)

    loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=False)
    line_mode = False
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)

    while loader.has_next():
        iter_info = loader.get_iterator_info()
        batch = loader.get_next()
        preprocessor.process_batch(batch)
        print(f'Batch: {iter_info[0]}/{iter_info[1]}')
    return


if __name__ == '__main__':
    main_fn()
