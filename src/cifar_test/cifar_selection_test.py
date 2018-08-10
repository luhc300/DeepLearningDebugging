from keras.datasets import cifar10
from src.cnn_profiler import CNNProfiler
from src.configs.network_configs.cifar.network_config_3 import NETWORK_STRUCTURE, NETWORK_ANCHOR, NETWORK_PATH, INIT, LEARNING_RATE
import numpy as np
import matplotlib.pyplot as plt
from src.distribution import Distribution
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32")
X_train /= 255
X_test = X_test.astype("float32")
X_test /= 255
def preprocess(x_train, x_test):
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])
    return x_train, x_test
X_train, X_test = preprocess(X_train, X_test)
cnn_profiler = CNNProfiler(NETWORK_STRUCTURE, network_anchor=NETWORK_ANCHOR, network_path=NETWORK_PATH, init=INIT, lr=LEARNING_RATE)
################## Train ######################
def train():
    x = X_train.astype("float32")
    print(X_train)
    y = y_train.reshape(-1)
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    cnn_profiler.train([None, 32, 32, 3], [None, 10], x, y, iter=6000)
################## Test ########################
def test():
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    cnn_profiler.test([None, 32, 32, 3], [None, 10], x[:2000], y[:2000])
train()