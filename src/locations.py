import os

# Filenames and paths to data.

# default value
model_dir = '../model'
fn_corpus = '../data/corpus.txt'

debug_directory = None
high_contrast = False
convert_fractions = False


def set_convert_fractions(convert_fractions0):
    global convert_fractions
    convert_fractions = convert_fractions0
    return


def get_convert_fractions():
    return convert_fractions


def set_high_contrast(high_contrast0):
    global high_contrast
    high_contrast = high_contrast0
    return


def get_high_contrast():
    return high_contrast


def set_model_dir(model_dir0):
    global model_dir
    model_dir = model_dir0
    return


def get_model_dir():
    return model_dir


def get_fn_char_list():
    return os.path.join(model_dir, 'charList.txt')


def get_fn_summary():
    return os.path.join(model_dir, 'summary.json')


def set_debug_dir(d_dir):
    global debug_directory
    debug_directory = d_dir
    if debug_directory:
        os.makedirs(debug_directory, exist_ok=True)
    return


def get_debug_dir():
    return debug_directory
