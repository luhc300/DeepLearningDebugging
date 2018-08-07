from keras.datasets import mnist
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from src.cnn_profiler import CNNProfiler
from src.configs.network_configs.mnist.network_config_1 import NETWORK_STRUCTURE, NETWORK_ANCHOR, NETWORK_PATH, INIT, LEARNING_RATE
from src.distribution import Distribution
import os
import numpy as np
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype("float32")
X_train /= 255
X_test = X_test.astype("float32")
X_test /= 255
cnn_profiler = CNNProfiler(NETWORK_STRUCTURE, network_anchor=NETWORK_ANCHOR, network_path=NETWORK_PATH, init=INIT, lr=LEARNING_RATE)

def train():
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    cnn_profiler.train([None, 28, 28, 1], [None, 10], x, y)
def test():
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    acc = 0
    for i in range(0, 8001, 2000):
        acc += cnn_profiler.test([None, 28, 28, 1], [None, 10], x[i:i+2000], y[i:i+2000])
    acc /= 5
    print(acc)

# mnist = read_data_sets("data/MNIST_data/", one_hot=True)
# x = mnist.train.images
# print(x[x>0])
# y = mnist.train.labels
# cnn_profiler.train([None, 28, 28, 1], [None, 10], x, y)
# x = mnist.test.images[:500]
# y = mnist.test.labels[:500]
# cnn_profiler.test([None, 28, 28, 1], [None, 10], x, y)


def test_m_distance(label):
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label]
    y = y[y == label]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    correct = cnn_profiler.get_correct_mid([None, 28, 28, 1], [None, 10], x, y)
    distribute_correct = Distribution(correct)
    own_dis = []
    for i in range(correct.shape[0]):
        dis = distribute_correct.mahanobis_distance(correct[i])
        own_dis.append(dis)
    own_dis = np.array(own_dis)
    # print(own_dis)
    print(own_dis.min())
    print(own_dis.max())
    print(own_dis.mean())
    xx = X_test.astype("float32")
    yy = y_test.reshape(-1)
    xx = xx[yy == label]
    yy = yy[yy == label]
    values = yy
    n_values = 10
    yy = np.eye(n_values)[values]
    correct_test = cnn_profiler.get_correct_mid([None, 28, 28, 1], [None, 10],  xx, yy)
    wrong_test = cnn_profiler.get_correct_mid([None, 28, 28, 1], [None, 10], xx, yy, wrong=True)
    test_correct_dis = []
    for i in range(correct_test.shape[0]):
        dis = distribute_correct.mahanobis_distance(correct_test[i])
        test_correct_dis.append(dis)
    test_correct_dis = np.array(test_correct_dis)
    test_wrong_dis = []
    for i in range(wrong_test.shape[0]):
        dis = distribute_correct.mahanobis_distance(wrong_test[i])
        test_wrong_dis.append(dis)
    test_wrong_dis = np.array(test_wrong_dis)
    # print(test_correct_dis)
    print(test_correct_dis.min())
    print(test_correct_dis.max())
    print(test_correct_dis.mean())
    # print(test_wrong_dis)
    print(test_wrong_dis.min())
    print(test_wrong_dis.max())
    print(test_wrong_dis.mean())
    x_another = X_train.astype("float32")
    y_another = y_train.reshape(-1)
    x_another = x_another[y_another == label+1]
    y_another = y_another[y_another == label+1]
    mid = cnn_profiler.get_mid([None, 28, 28, 1], [None, 10], x_another)
    another_dis = []
    for i in range(mid.shape[0]):
        dis = distribute_correct.mahanobis_distance(mid[i])
        another_dis.append(dis)
    another_dis = np.array(another_dis)
    # print(another_dis)
    print(another_dis.min())
    print(another_dis.max())
    print(another_dis.mean())

def test_with_filter(label):
    pass
test_m_distance(5)
