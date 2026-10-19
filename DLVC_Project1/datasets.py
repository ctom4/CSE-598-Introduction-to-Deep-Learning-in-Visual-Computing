import os
import numpy as np

# Folder where this datasets.py file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Since your CSV files are in the same folder as datasets.py
datapath = BASE_DIR


def dataset(noTrSamples, noTsSamples, loaded, Normalize):
    np.random.seed(1)
    loaded = np.nan_to_num(loaded)

    tr_index = np.random.choice(loaded.shape[0], noTrSamples, replace=False)
    ind = np.zeros(loaded.shape[0], dtype=bool)
    ind[tr_index] = True

    trX = loaded[ind, :-1]
    trY = loaded[ind, -1]

    tsX = loaded[~ind, :-1]
    tsY = loaded[~ind, -1]

    # Normalize the data
    if Normalize:
        trX = np.nan_to_num(trX / trX.max(axis=0))
        tsX = np.nan_to_num(tsX / tsX.max(axis=0))

    return trX, trY, tsX, tsY


def ridge_reg_data(noTrSamples=300, noTsSamples=100):
    fd = os.path.join(datapath, 'ridge_signal.csv')
    loaded = np.genfromtxt(fname=fd, skip_header=1, delimiter=',')
    return dataset(noTrSamples, noTsSamples, loaded, Normalize=False)


def pima_data(noTrSamples=500, noTsSamples=268):
    fd = os.path.join(datapath, 'diabetes.csv')
    loaded = np.genfromtxt(fname=fd, skip_header=1, delimiter=',')
    return dataset(noTrSamples, noTsSamples, loaded, Normalize=False)
