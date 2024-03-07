#!/usr/bin/env python3

import argparse
import json
import os.path

import cv2

import locations
from model import Model, DecoderType
from main import char_list_from_file
from preprocessor import Preprocessor
from dataloader_iam import Batch

# not using line_mode or wide_mode
PREPROCESSOR_SIZE = (128, 32)


def parse_args() -> argparse.Namespace:
    """
    Parse arguments from the command line.
    """
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


def main():
    """
    Load the model and apply it to everything in the control file.
    :return:
    """
    args = parse_args()
    locations.set_model_dir(args.model_dir)
    decoder_type = DecoderType.BestPath
    model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
    apply_all(model, args.control_file, args.output_dir)
    return


def apply_all(model: Model, control_file: str, output_dir: str):
    """
    Apply the model to the specified images.
    :param model:
    :param control_file:
    :param output_dir:
    :return:
    """
    input_dir = os.path.dirname(os.path.realpath(control_file))
    # not using line_mode or wide_mode
    preprocessor = Preprocessor(PREPROCESSOR_SIZE)
    with open(control_file, 'r') as f:
        control = json.load(f)
    for entry in control:
        apply(model, preprocessor, input_dir, entry, output_dir)
    return


def apply(model: Model, preprocessor: Preprocessor, control_path: str, entry: dict, output_dir: str):
    """
    Apply the model to the specified image and save the output.
    :param model:
    :param preprocessor:
    :param control_path:
    :param entry:
    :param output_dir:
    :return:
    """
    image_file = entry['input_file']
    image_path = os.path.join(control_path, image_file)
    image = locations.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # HS original code uses dynamic width and padding (unlike training)
    # preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    # This is more cowbell:
    image = preprocessor.process_img(image)
    batch = Batch([image], None, 1)
    text, probability = model.infer_batch(batch, True)
    output = {'text': text,
              'probability': probability,
              'input_path': image_path}
    output.update(entry)
    output_file = f'{entry["identifier"]}.json'
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(output, f)
    return


if __name__ == '__main__':
    main()
