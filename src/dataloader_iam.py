import os.path
import pickle
import random
from collections import namedtuple
from typing import Tuple

import cv2
import lmdb
import numpy as np
from path import Path
import locations

Sample = namedtuple('Sample', ['gt_text', 'file_path'])
Batch = namedtuple('Batch', ['imgs', 'gt_texts', 'batch_size'])


class DataLoaderIAM:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 data_split: float = 0.95,
                 fast: bool = True) -> None:
        """Loader for dataset."""

        assert data_dir.exists()

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        f = open(data_dir / 'words.txt')
        chars = set()
        for line in f:
            # ignore empty and comment lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            line_split = line.split(' ', maxsplit=2)
            assert len(line_split) == 2

            # columns:
            # 0 filename
            # 1 word(s)

            local_filename = line_split[0]
            gt_text = line_split[1]

            file_name = data_dir / 'img' / local_filename

            # GT text are columns starting at 9
            gt_text = ' '.join(line_split[8:])
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
            if not os.path.exists(img_path):
                print(f'{img_path} is missing')
                return None
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if locations.get_debug_dir():
                if img is None:
                    print(f'{img_path} is None')
                else:  # None
                    print(f'{img_path} is good')
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
