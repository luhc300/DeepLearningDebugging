from keras.datasets import mnist
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from src.cnn_profiler import CNNProfiler
from src.configs.network_configs.mnist.network_config_4 import NETWORK_STRUCTURE, NETWORK_ANCHOR, NETWORK_PATH, INIT, LEARNING_RATE
from src.distribution import Distribution
import matplotlib.pyplot as plt
import src.augmentation as aug
import os
import numpy as np
(X_train, y_train), (X_test, y_test) = mnist.load_data()
test_x, test_y= aug.random_pertubate(X_train[:2], y_train[:2], 1)
# plt.subplot(211)
# plt.imshow(X_train[0].reshape(28,28))
# plt.subplot(212)
# plt.imshow(test_x[0].reshape(28,28))
# plt.show()
X_ptb, y_ptb = aug.random_pertubate(X_train, y_train, 1)
X_train = np.concatenate((X_train[:1000], X_ptb), axis=0)
y_train = np.concatenate((y_train[:1000], y_ptb), axis=0)
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
    cnn_profiler.train([None, 28, 28, 1], [None, 10], x, y, iter=4000)
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
def small_rate(data, range):
    data_in_range = data[data <= range]
    rate = len(data_in_range) / len(data)
    return rate

def test_m_distance(label):
    anchor = -2
    filter = None
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label]
    y = y[y == label]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    correct = cnn_profiler.get_correct_mid([None, 28, 28, 1], [None, 10], x, y, anchor=anchor, filter=filter)
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
    thres = own_dis.mean() * 2
    xx = X_test.astype("float32")
    yy = y_test.reshape(-1)
    xx = xx[yy == label]
    yy = yy[yy == label]
    values = yy
    n_values = 10
    yy = np.eye(n_values)[values]
    correct_test = cnn_profiler.get_correct_mid([None, 28, 28, 1], [None, 10],  xx, yy, anchor=anchor, filter=filter)
    wrong_test = cnn_profiler.get_correct_mid([None, 28, 28, 1], [None, 10], xx, yy, wrong=True, anchor=anchor, filter=filter)
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
    print(small_rate(test_correct_dis, thres))
    # print(test_wrong_dis)
    print(test_wrong_dis.min())
    print(test_wrong_dis.max())
    print(test_wrong_dis.mean())
    print(small_rate(test_wrong_dis, thres))
    x_another = X_test.astype("float32")
    y_another = y_test.reshape(-1)
    x_another = x_another[y_another != label]
    y_another = y_another[y_another != label]
    values = y_another
    n_values = 10
    y_another = np.eye(n_values)[values]
    mid = cnn_profiler.get_correct_mid([None, 28, 28, 1], [None, 10], x_another, y_another, anchor=anchor, filter=filter)
    another_dis = []
    for i in range(mid.shape[0]):
        dis = distribute_correct.mahanobis_distance(mid[i])
        another_dis.append(dis)
    another_dis = np.array(another_dis)
    # print(another_dis)
    print(another_dis.min())
    print(another_dis.max())
    print(another_dis.mean())
    print(small_rate(another_dis, thres))


def test_with_filter(label):
    anchor=-2
    filter=None
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label]
    y = y[y == label]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    correct = cnn_profiler.get_correct_mid([None, 28, 28, 1], [None, 10], x, y, anchor=anchor,filter=filter)
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
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    x = x[y != label][:2000]
    y = y[y != label][:2000]
    wrong_test, pic = cnn_profiler.get_mid_by_label([None, 28, 28, 1], [None, 10],  x, y, label, anchor=anchor,filter=filter)
    logits, pic = cnn_profiler.get_mid_by_label([None, 28, 28, 1], [None, 10],  x, y, label, anchor=-1)
    pic = pic.reshape(28,28)
    plt.imshow(pic)
    plt.show()
    print(logits)
    test_wrong_dis = []
    for i in range(wrong_test.shape[0]):
        dis = distribute_correct.mahanobis_distance(wrong_test[i])
        test_wrong_dis.append(dis)
    test_wrong_dis = np.array(test_wrong_dis)
    # print(test_correct_dis)
    print(test_wrong_dis)
    print(test_wrong_dis.min())
    print(test_wrong_dis.max())
    print(test_wrong_dis.mean())
test_m_distance(8)
