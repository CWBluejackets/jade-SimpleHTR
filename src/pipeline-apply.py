#!/usr/bin/env python3

from typing import Tuple

import numpy as np

import locations
from dataloader_iam import Batch
from main import char_list_from_file
from model import Model, DecoderType
from preprocessor import Preprocessor

# not using line_mode or wide_mode
PREPROCESSOR_SIZE = (128, 32)

model = None
decoder_type = DecoderType.BestPath
preprocessor = Preprocessor(PREPROCESSOR_SIZE)


def set_model(model_dir):
    global model
    locations.set_model_dir(model_dir)
    model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=False)
    return


def apply(image: np.ndarray) -> Tuple[str, float]:
    if model is None:
        raise RuntimeError("Model is not set")
    image = preprocessor.process_img(image)
    batch = Batch([image], None, 1)
    text, probability = model.infer_batch(batch, True)
    return text, probability
