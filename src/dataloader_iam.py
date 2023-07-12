import os.path
import pickle
import random
import re
from collections import namedtuple
from typing import Tuple

import cv2
import lmdb
import numpy as np
from path import Path
import locations
import tensorflow as tf

Sample = namedtuple('Sample', ['gt_text', 'file_path'])
Batch = namedtuple('Batch', ['imgs', 'gt_texts', 'batch_size'])

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
    img = tf.io.read_file(os.path.join(workflow_dir, img_path))
    # TODO check channels=1
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.resize(img, size=(100, 100)) # optional
    label1 = tf.cast(tf.equal(label, 'class2'), tf.int32)
    return img, label1


def load_dataset(control_path):
    # TODO use:
    locations.get_high_contrast()
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


class DataLoaderIAM:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self,
                 data_file: Path,
                 batch_size: int,
                 data_split: float = 0.95,
                 fast: bool = True) -> None:
        """Loader for dataset."""

        assert data_file.exists()
        data_dir = data_file.dirname()

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        chars = set()

        with open(data_file) as f:
            for line in f:
                # ignore empty and comment lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # maxsplit is max number of cuts, not number of pieces
                line_split = line.split(' ', maxsplit=1)
                # columns:
                # 0 filename
                # 1 word(s)

                local_filename = line_split[0]
                file_name = data_dir / local_filename
                # sometimes the training data includes a correct empty string
                if len(line_split) == 2:
                    gt_text = line_split[1]
                    if locations.get_convert_fractions():
                        gt_text = convert_fractions(gt_text)
                else:
                    # CTLoss doesn't accept zero-length labels
                    gt_text = ' '
                chars = chars.union(set(list(gt_text)))

                # put sample into list if file is correct
                if os.path.exists(file_name):
                    self.samples.append(Sample(gt_text, file_name))
                else:
                    print(f'File missing: {file_name}')

        # split into training and validation set: 95% - 5%
        split_idx = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        # put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    def train_set(self) -> None:
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller

    def _get_img(self, i: int) -> np.ndarray:
        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img_path = self.samples[i].file_path
            img = locations.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if locations.get_high_contrast():
                img = cv2.normalize(img, 0, 255, cv2.NORM_MINMAX)
            if locations.get_debug_dir():
                if img is None:
                    print(f'{img_path} is None')
                else:  # None
                    output_filename = os.path.join(locations.get_debug_dir(),
                                                   os.path.basename(img_path) + '_test.png')
                    cv2.imwrite(output_filename, img)
        return img

    def get_next(self) -> Batch:
        """Get next element."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))
