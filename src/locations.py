import os

# Filenames and paths to data.

# default value
model_dir = '../model'

fn_corpus = '../data/corpus.txt'

debug_directory = None


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
