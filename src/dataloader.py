import os
import random
import re
from typing import List, Tuple, Union

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2

import locations
import model

substitutions = [(re.compile(r'\b1/4\b'), '¼'),
                 (re.compile(r'\b1/2\b'), '½'),
                 (re.compile(r'\b3/4\b'), '¾')
                 ]


def convert_fractions(text):
    for pattern, replacement in substitutions:
        text = pattern.sub(replacement, text)
    return text


def get_png_tensor(png_path) -> tf.Tensor:
    """
    part of the tf data loading
    :return:
    """
    raw = tf.io.read_file(png_path)
    # channels = 1 -> grayscale
    image = tf.image.decode_png(raw, channels=1)
    if locations.get_high_contrast():
        image = tf.image.adjust_contrast(image, 10)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize; maintain aspect ratio; pad with zeroes
    # TODO This may not be necessary? if using keras.preprocessing.sequence.pad(...) later
    # Maybe it is needed so tf.stack _x keeps working.
    image = tf.image.resize_with_pad(image, model.INPUT_SHAPE[0], model.INPUT_SHAPE[1])
    # -> tf.float32
    return image


def get_png_and_label(workflow_dir, path_and_label, max_length, character_list) -> tuple:
    """
    part of the tensorflow data loading
    :param path_and_label: tuple (path, label)
    :param workflow_dir: .../training_data/21955
    :return:
    """
    img_full_path = os.path.join(workflow_dir, path_and_label[0])
    image_t = get_png_tensor(img_full_path)
    label = path_and_label[1]
    padding = [0] * (max_length - len(label))
    label_array = [character_list.index(c) for c in label] + padding
    # label_t = tf.strings.unicode_decode(label, input_encoding='utf-8')
    label_t = tf.convert_to_tensor(label_array, dtype=tf.int32)
    return image_t, label_t


def load_raw_dataset(control_file_path) -> List[tuple]:
    """
    Load the PNG files and labels to a list of pairs of tensors
    :param control_file_path: words file
    :return: list of pairs of tensors (image, label)
    """
    control_data = []
    with open(control_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line and not line.startswith('#'):
                img_path, label = line.split(' ', maxsplit=1)
                if locations.get_convert_fractions():
                    label = convert_fractions(label)
                control_data.append((img_path, label))
    max_length, character_list = analyse_labels(control_data)
    workflow_dir = os.path.dirname(control_file_path)
    x_y_all = [get_png_and_label(workflow_dir, path_and_label, max_length, character_list)
               for path_and_label in control_data]
    return x_y_all, character_list


def analyse_labels(control_data: List[Tuple[str, str]]) -> Tuple[int, List[str]]:
    max_length = 0
    # the TAB character will get sorted to 0 so not used
    character_set = {'\t', ' '}
    for path, label in control_data:
        max_length = max(max_length, len(label))
        character_set |= set(label)
    return max_length, sorted(character_set)


def load_dataset(control_file_path, validation_split: float) -> tuple[
        Union[DatasetV1, DatasetV2], Union[DatasetV1, DatasetV2], List[str]]:
    """
    Load the PNG files and labels as tensors and structure for ease of use
    :param validation_split: proportion of data to set aside
    :param control_file_path: words file
    :return: tensorflow Dataset
    """
    if (validation_split < 0.0) or (validation_split > 1.0):
        raise ValueError('split out of range')
    x_y_all, character_list = load_raw_dataset(control_file_path)
    random.shuffle(x_y_all)
    # we have a list of tuples of tensors (image, label)
    cutoff = round(len(x_y_all) * validation_split)
    validation_pairs = x_y_all[:cutoff]
    training_pairs = x_y_all[cutoff:]

    # pick up batch size from command line arguments
    validation_ds = tf.data.experimental.from_list(validation_pairs).batch(locations.get_batch_size())
    training_ds = tf.data.experimental.from_list(training_pairs).batch(locations.get_batch_size())
    return training_ds, validation_ds, character_list


def tensor_to_string(tensor: tf.Tensor, character_list: List[str]) -> str:
    # ignore zeroes
    characters = [character_list[i] for i in tensor.numpy() if i > 0]
    return ''.join(characters)

# alternative ideas
# https://medium.com/the-owl/creating-a-tf-dataset-using-a-data-generator-5e5564609e64

# https://tensorexamples.com/2020/07/27/Using-the-tf.data.Dataset.html
# Method 2: Using a generator
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
