import argparse
import json
from typing import Tuple, List

from path import Path

import locations
from model import Model


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False, wide_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode or wide_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(locations.get_fn_summary(), 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(locations.get_fn_char_list()) as f:
        return list(f.read())

    # TODO reuse this stuff
    # df = pd.DataFrame.from_records(results, columns=['file', 'label', 'output', 'probability', 'similarity',
    #                                                  'correct', 'distance', 'length'])
    # df['logprob'] = df['probability'].apply(math.log1p)
    # print()
    # print(str(df.describe()))
    # print()
    # print('PEARSON')
    # print(str(df[['probability', 'logprob', 'similarity', 'distance', 'length']].corr('pearson')))
    # print()
    # character_error = df['distance'].sum() / df['length'].sum()
    # print('CHARACTER ERROR')
    # print(f'{character_error}')
    # print(f'{character_error:.1%}')
    # print()
    # word_accuracy = df['correct'].value_counts().get(True, 0) / df['correct'].size
    # print('WORD ACCURACY')
    # print(f'{word_accuracy}')
    # print(f'{word_accuracy:.1%}')
    # print()
    # if output_file:
    #     print(f'Writing to {output_file}')
    #     df.to_csv(output_file)
    # return


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer', 'apply'],
                        default='infer')
    parser.add_argument('--batch_size', help='Batch size.',
                        type=int, default=100)
    parser.add_argument('--data_file', help='words file pointing to images',
                        type=Path, required=False)
    # TODO implement augment
    parser.add_argument('--augment', help=' Use data augmentation',
                        default=False, action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.',
                        type=Path, default='../data/word.png')
    parser.add_argument('--model_dir', help='Path to trained model',
                        type=Path, default='../model')
    parser.add_argument('--convert_fractions', help='convert 1/4 1/2 3/4 to single unicode characters',
                        action='store_true', default=False)
    parser.add_argument('--high_contrast', help='normalize image matrix to maximum contrast',
                        action='store_true', default=False)
    parser.add_argument('--output_file', help='output file for apply mode',
                        type=Path, default=None)
    # possibly obsolete
    parser.add_argument('--early_stopping', help='Early stopping epochs.',
                        type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).',
                        action='store_true', default=False)
    parser.add_argument('--debug_dir', help='directory to dump CV2',
                        type=Path, default=None)

    return parser.parse_args()


def main():
    """Main function."""

    # parse arguments and set model location and CTC decoder
    args = parse_args()
    locations.set_model_dir(args.model_dir)
    locations.set_debug_dir(args.debug_dir)
    locations.set_high_contrast(args.high_contrast)
    locations.set_convert_fractions(args.convert_fractions)
    locations.set_batch_size(args.batch_size)

    # create empty model
    the_model = Model()

    # train the model
    if args.mode == 'train':
        the_model.train(args.data_file, 0.05)
        the_model.save()

    # evaluate it on the validation set
    elif args.mode == 'validate':
        the_model.load()
        prediction_s, metrics = the_model.evaluate(args.data_file)
        # TODO sensible output
        print(prediction_s)
        print(metrics)

    # infer text from test image
    elif args.mode == 'infer':
        the_model.load()
        prediction = the_model.apply(args.img_file)
        print(prediction)

    elif args.mode == 'apply':
        the_model.load()
        the_model.load()
        prediction_s, metrics = the_model.evaluate(args.data_file)
        # TODO sensible output
        print(prediction_s)
        print(metrics)
        # apply(the_model, args.data_file, args.output_file)


if __name__ == '__main__':
    main()
