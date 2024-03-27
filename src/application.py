#!/usr/bin/env python3

from typing import Tuple

import numpy as np

from dataloader_iam import Batch
from model_nonglobal import Model
from preprocessor import Preprocessor

# not using line_mode or wide_mode
PREPROCESSOR_SIZE = (128, 32)


class TrainedModel:

    def __init__(self, model_directory):
        self.model_directory = model_directory
        self.model = Model(self.model_directory)
        self.preprocessor = Preprocessor(PREPROCESSOR_SIZE)
        return

    # TODO proper big batches

    def apply(self, image: np.ndarray) -> Tuple[str, float]:
        image = self.preprocessor.process_img(image)
        batch = Batch([image], None, 1)
        text, probability = self.model.infer_batch(batch, True)
        return text, probability
