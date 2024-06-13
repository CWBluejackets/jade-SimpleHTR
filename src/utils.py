import json

import numpy as np

# used for quantile debugging
EIGHTHS = [x / 8 for x in range(0, 9)]

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
