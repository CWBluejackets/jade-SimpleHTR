# dataloader based partly on
# https://medium.com/the-owl/creating-a-tf-dataset-using-a-data-generator-5e5564609e64
import copy
import os
import random
import re

import numpy as np
import tensorflow as tf

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


def get_png_and_label(workflow_dir, img_path, label):
    """
    part of the tf data loading
    :param workflow_dir: .../training_data/21955
    :param img_path: img/390403.png
    :param label: Smith Robt
    :return:
    """
    raw = tf.io.read_file(os.path.join(workflow_dir, img_path))
    # channels = 1 -> grayscale
    image = tf.image.decode_png(raw, channels=1)
    if locations.get_high_contrast():
        image = tf.image.adjust_contrast(image, 10)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize; maintain aspect ratio; pad with zeroes
    # this may not be necessary? if using keras.preprocessing.sequence.pad(...) later
    image = tf.image.resize_with_pad(image, model.INPUT_SHAPE[0], model.INPUT_SHAPE[1])
    # tensor shape is length of string
    # label1 = tf.convert_to_tensor(list(label))
    # or is numeric better?
    label_t = tf.strings.unicode_decode(label, input_encoding='utf-8')
    # Note for later: encoding has to be done with uc UTF-8
    # tf.strings.unicode_encode(t0, output_encoding='UTF-8')
    return image, label_t


def load_dataset(control_path):
    # https://stackoverflow.com/questions/63640185/make-a-tf-dataset-from-image-filenames-and-corresponding-string-labels
    control_data = []
    with open(control_path, 'r') as f:
        for line in f.readlines():
            img_path, label = line.strip('\n').split(' ', maxsplit=1)
            if locations.get_convert_fractions():
                label = convert_fractions(label)
            control_data.append([img_path, label])
    workflow_dir = os.path.dirname(control_path)
    dataset = tf.data.Dataset.from_tensor_slices(control_data).map(lambda x:
                                                                   get_png_and_label(workflow_dir, x[0], x[1]))
    return dataset

# TODO
# Initialization:
# parameters: control file, train/test split
# read the control file, set the size, store list of paths and list of labels;
# shuffle list and do the split.
# Also needed: list of input lengths and label lengths (for CTCModel)
# Generator/iterator:
#



class ArrayDataset:
    def __init__(self,
                 phase,
                 array,
                 labels,
                 mean, std,
                 transformations=None):
        self.phase = phase
        self.imgarr = copy.deepcopy(array)
        self.labels = copy.deepcopy(labels)
        self.mean = mean
        self.std = std
        self.transforms = transformations

    def __len__(self):
        return self.imgarr.shape[0]

    def __getitem__(self, idx):
        x = self.imgarr[idx]
        x = self.augment(x)
        y = self.labels[idx].astype(np.int64)
        return x, y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def normalize(self, x):
        return (x - self.mean) / self.std

    # shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        reidx = random.sample(population=list(range(self.__len__())), k=self.__len__())
        self.imgarr = self.imgarr[reidx]
        self.labels = self.labels[reidx]

    # applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, x):
        if self.transforms is not None:
            x = self.transforms(x)
        x = self.normalize(x)
        return x
