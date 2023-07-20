import os
import random
import re
from typing import List

import tensorflow as tf
from keras.preprocessing import sequence

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


def get_png_and_label(workflow_dir, path_and_label):
    """
    part of the tf data loading
    :param path_and_label: tuple (path, label)
    :param workflow_dir: .../training_data/21955
    :return:
    """
    img_full_path = os.path.join(workflow_dir, path_and_label[0])
    raw = tf.io.read_file(img_full_path)
    # channels = 1 -> grayscale
    image = tf.image.decode_png(raw, channels=1)
    if locations.get_high_contrast():
        image = tf.image.adjust_contrast(image, 10)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize; maintain aspect ratio; pad with zeroes
    # TODO This may not be necessary? if using keras.preprocessing.sequence.pad(...) later
    image = tf.image.resize_with_pad(image, model.INPUT_SHAPE[0], model.INPUT_SHAPE[1])
    # -> tf.float32
    label = path_and_label[1]
    label_t = tf.strings.unicode_decode(label, input_encoding='utf-8')
    # -> tf.int32; shape is length of string
    return image, label_t


def load_raw_dataset(control_file_path) -> List[tuple]:
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


def load_unpadded_dataset(control_file_path) -> tuple:
    x_y_all = load_raw_dataset(control_file_path)
    random.shuffle(x_y_all)
    x_train = [pair[0] for pair in x_y_all]
    y_train = [pair[1] for pair in x_y_all]
    return x_train, y_train


def load_dataset_for_ctc(control_file_path) -> tuple:
    x_train, y_train = load_unpadded_dataset(control_file_path)
    x_train_len = [len(item) for item in x_train]
    y_train_len = [len(item) for item in y_train]
    # image padding done already for now
    x_train_pad = x_train
    # string padding needed
    y_train_pad = sequence.pad_sequences(y_train, padding='post', value=0, dtype=tf.int32)
    return x_train_pad, x_train_len, y_train_pad, y_train_len


def tensor_to_string(tensor: tf.Tensor) -> str:
    if tensor.dtype != tf.string:
        raise ValueError(f'tensor content {tensor.dtype} is not string')
    return tensor.numpy().decode('utf-8')


# alternative ideas
# https://medium.com/the-owl/creating-a-tf-dataset-using-a-data-generator-5e5564609e64

# https://tensorexamples.com/2020/07/27/Using-the-tf.data.Dataset.html
# Method 2: Using a generator
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
