import pickle 
import numpy as np
from glob import glob
import os

def load_batches(filename):
    with open(filename, 'rb') as fn:
        dict = pickle.load(fn, encoding='latin1')
    return dict
def get_X(dict):
    return dict['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
def get_y(dict):
    return dict['labels']
def preprocess(file):
    X = list()
    y = list()
    for batch_file in file:
        dict = load_batches(batch_file)
        X.append(get_data(dict))
        y += (get_label(dict))
    X = np.concatenate(X, axis=0).astype(np.float32)
    X /= 255
    y = np.array(y)
    return X, y
def preprocess_train(dir_name):
    return preprocess(glob(os.path.join(dir_name, '*_batch_*')))
def preprocess_test(dir_name):
    return preprocess(glob(os.path.join(dir_name, 'test_batch')))
