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


def get_png_and_label(workflow_dir, path_and_label) -> tuple:
    """
    part of the tensorflow data loading
    :param path_and_label: tuple (path, label)
    :param workflow_dir: .../training_data/21955
    :return:
    """
    img_full_path = os.path.join(workflow_dir, path_and_label[0])
    image_t = get_png_tensor(img_full_path)
    label = path_and_label[1]
    label_t = tf.strings.unicode_decode(label, input_encoding='utf-8')
    # -> tf.int32; shape is length of string
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
    workflow_dir = os.path.dirname(control_file_path)
    x_y_all = [get_png_and_label(workflow_dir, path_and_label) for path_and_label in control_data]
    return x_y_all


def load_dataset(control_file_path, validation_split: float) -> tuple[
        Union[DatasetV1, DatasetV2], Union[DatasetV1, DatasetV2]]:
    """
    Load the PNG files and labels as tensors and structure for ease of use
    :param validation_split: proportion of data to set aside
    :param control_file_path: words file
    :return: tensorflow Dataset
    """
    if (validation_split < 0.0) or (validation_split > 1.0):
        raise ValueError('split out of range')
    x_y_all = load_raw_dataset(control_file_path)
    random.shuffle(x_y_all)
    # we have a list of tuples of tensors (image, label)
    cutoff = round(len(x_y_all) * validation_split)
    validation_pairs = x_y_all[:cutoff]
    training_pairs = x_y_all[cutoff:]
    validation_ds = tf.data.experimental.from_list(validation_pairs)
    training_ds = tf.data.experimental.from_list(training_pairs)
    return training_ds, validation_ds


def tensor_to_string(tensor: tf.Tensor) -> str:
    # TODO change to corpus-based charset indexing
    if tensor.dtype != tf.string:
        raise ValueError(f'tensor content {tensor.dtype} is not string')
    return tensor.numpy().decode('utf-8')

# alternative ideas
# https://medium.com/the-owl/creating-a-tf-dataset-using-a-data-generator-5e5564609e64

# https://tensorexamples.com/2020/07/27/Using-the-tf.data.Dataset.html
# Method 2: Using a generator
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
