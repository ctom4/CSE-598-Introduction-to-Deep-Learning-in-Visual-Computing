import os
import gzip
import numpy as np
from urllib.request import urlretrieve
from urllib.parse import urljoin
import random
from scipy import io
import bz2

random.seed(1)
np.random.seed(1)

# Dataset download path
mnist_url = 'http://yann.lecun.com/exdb/mnist/'
svhn_url = 'http://ufldl.stanford.edu/housenumbers/'
mnist_m_url = 'https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz'
usps_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/'

cwd = os.getcwd()

data_root = os.path.join(cwd, 'zylabfiles')
data_dir = os.path.join(data_root, 'data')

if not os.path.isdir(data_root):
    os.mkdir(data_root)

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

datapath = data_dir

def get_mnist(getRGB=False, setSizeTo32=False):
    trainFileName = 'train-images-idx3-ubyte.gz'
    trainLabelsFileName = 'train-labels-idx1-ubyte.gz'
    testFileName = 't10k-images-idx3-ubyte.gz'
    testLabelsFileName = 't10k-labels-idx1-ubyte.gz'

    # Download and extract
    download_extract_mnist(datapath, mnist_url, trainFileName)
    download_extract_mnist(datapath, mnist_url, testFileName)
    download_extract_mnist(datapath, mnist_url, trainLabelsFileName)
    download_extract_mnist(datapath, mnist_url, testLabelsFileName)

    # Load training data
    with open(os.path.join(datapath, trainFileName.split('.gz')[0]), 'rb') as fd:
        loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 1, 28, 28)).astype(np.float32)

    with open(os.path.join(datapath, trainLabelsFileName.split('.gz')[0]), 'rb') as fd:
        loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(np.int64)

    # Load test data
    with open(os.path.join(datapath, testFileName.split('.gz')[0]), 'rb') as fd:
        loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 1, 28, 28)).astype(np.float32)

    with open(os.path.join(datapath, testLabelsFileName.split('.gz')[0]), 'rb') as fd:
        loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(np.int64)

    # Optional padding
    if setSizeTo32:
        trData = np.pad(trData, ((0,0),(0,0),(2,2),(2,2)), 'constant')
        tsData = np.pad(tsData, ((0,0),(0,0),(2,2),(2,2)), 'constant')

    # Optional RGB
    if getRGB:
        trData = np.repeat(trData, 3, axis=1)
        tsData = np.repeat(tsData, 3, axis=1)

    # Normalize [-1, 1]
    trData = 2 * (trData / 255) - 1
    tsData = 2 * (tsData / 255) - 1

    trLabels = trLabels.squeeze()
    tsLabels = tsLabels.squeeze()

    return trData, trLabels, tsData, tsLabels


def get_svhn(getRGB=False):
    trainFileName = 'train_32x32.mat'
    testFileName = 'test_32x32.mat'

    download_extract(datapath, svhn_url, trainFileName)
    download_extract(datapath, svhn_url, testFileName)

    svhn_tr_data = io.loadmat(os.path.join(datapath, trainFileName))
    trLabels = svhn_tr_data['y'].astype(np.int64).squeeze()
    np.place(trLabels, trLabels == 10, 0)

    trData = svhn_tr_data['X'].astype(np.float32)

    if getRGB:
        trData = trData.transpose(3,2,0,1)
    else:
        trData = (trData[:,:,0,:]*0.2125 + trData[:,:,1,:]*0.7154 + trData[:,:,2,:]*0.0721)
        trData = trData.reshape(32,32,1,-1).transpose(3,2,0,1)

    trData = 2 * (trData / 255) - 1

    svhn_ts_data = io.loadmat(os.path.join(datapath, testFileName))
    tsLabels = svhn_ts_data['y'].astype(np.int64).squeeze()
    np.place(tsLabels, tsLabels == 10, 0)

    tsData = svhn_ts_data['X'].astype(np.float32)

    if getRGB:
        tsData = tsData.transpose(3,2,0,1)
    else:
        tsData = (tsData[:,:,0,:]*0.2125 + tsData[:,:,1,:]*0.7154 + tsData[:,:,2,:]*0.0721)
        tsData = tsData.reshape(32,32,1,-1).transpose(3,2,0,1)

    tsData = 2 * (tsData / 255) - 1

    return trData, trLabels, tsData, tsLabels


def get_usps(getRGB=False):
    trainFileName = 'usps.bz2'
    testFileName = 'usps.t.bz2'

    download_extract(datapath, usps_url, trainFileName)
    download_extract(datapath, usps_url, testFileName)

    with bz2.open(os.path.join(datapath, trainFileName)) as fp:
        raw_data = [line.decode().split() for line in fp.readlines()]
        trData = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
        trData = np.asarray(trData, dtype=np.float32).reshape((-1, 1, 16, 16))
        trLabels = np.asarray([int(d[0]) - 1 for d in raw_data], dtype=np.int64)

    with bz2.open(os.path.join(datapath, testFileName)) as fp:
        raw_data = [line.decode().split() for line in fp.readlines()]
        tsData = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
        tsData = np.asarray(tsData, dtype=np.float32).reshape((-1, 1, 16, 16))
        tsLabels = np.asarray([int(d[0]) - 1 for d in raw_data], dtype=np.int64)

    if getRGB:
        trData = np.repeat(trData, 3, axis=1)
        tsData = np.repeat(tsData, 3, axis=1)

    return trData, trLabels, tsData, tsLabels


def download_extract(datapath, url, filename):
    full_path = os.path.join(datapath, filename)
    if not os.path.exists(full_path):
        url = urljoin(url, filename)
        urlretrieve(url, full_path)


def download_extract_mnist(datapath, url, filename):
    full_path = os.path.join(datapath, filename)
    extract_full_path = os.path.join(datapath, filename.split('.gz')[0])

    if os.path.exists(extract_full_path):
        return

    if not os.path.exists(full_path):
        url = urljoin(url, filename)
        urlretrieve(url, full_path)

    with gzip.open(full_path, 'rb') as input_file:
        data = input_file.read()

    with open(extract_full_path, 'wb') as output_file:
        output_file.write(data)