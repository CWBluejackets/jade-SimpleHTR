import json
import os

import cv2
import numpy as np

# used for quantile output
OCTILES = [x / 8 for x in range(0, 9)]
QUARTILES = [x / 8 for x in range(0, 5)]

# the cv2 color scheme is (blue, green, red)
RED = (0, 0, 255)
GREEN = (0, 128, 0)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)


class NumpyJsonEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    # custom encoder to get numpy types through
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(NumpyJsonEncoder, self).default(obj)


def safe_imread(img_path, cv2_flags, exception_if_missing=False):
    if os.path.exists(img_path):
        result = cv2.imread(img_path, cv2_flags)
    else:
        result = None
        message = f'File {img_path} is missing!'
        if exception_if_missing:
            raise FileNotFoundError(message)
        else:
            print(message)
    return result


def write_sheet_info(output_dir, output_prefix, sheet_info):
    sheet_info_file = os.path.join(output_dir, f'{output_prefix}.json')
    print(f'SUMMARY: Writing {sheet_info_file}')
    with open(sheet_info_file, 'w') as f:
        json.dump(sheet_info, f, indent=1, sort_keys=True, cls=NumpyJsonEncoder)
    return

