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
    cnn_profiler.train([None, 32, 32, 3], [None, 10], x, y, iter=2500)
################## Test ########################
def test():
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    cnn_profiler.test([None, 32, 32, 3], [None, 10], x[:2000], y[:2000])
def get_distribution(label, anchor, filter):
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label]
    y = y[y == label]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    correct = cnn_profiler.get_correct_mid([None, 32, 32, 3], [None, 10], x, y, anchor=anchor, filter=filter)
    distribute_correct = Distribution(correct)
    return distribute_correct
def calc_distribution(anchor, filter):
    distribution_list = []
    for i in range(10):
        distribution_list.append(get_distribution(i, anchor, filter))
    return distribution_list
def test_m_dist(label):
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label]
    y = y[y == label]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    anchor = -2
    filter = None
    test_correct_dis = []
    correct_test = cnn_profiler.get_correct_mid([None, 32, 32, 3], [None, 10], x, y, anchor=anchor, filter=filter)
    wrong_test = cnn_profiler.get_correct_mid([None, 32, 32, 3], [None, 10], x, y, wrong=True, anchor=anchor, filter=filter)
    distribution_list = calc_distribution(anchor, filter)
    for j in range(10):
        distribute_correct = distribution_list[j]
        for i in range(correct_test.shape[0]):
            dis = distribute_correct.mahanobis_distance(correct_test[i])
            test_correct_dis.append(dis)
        test_correct_dis = np.array(test_correct_dis)
        test_wrong_dis = []
        for i in range(wrong_test.shape[0]):
            dis = distribute_correct.mahanobis_distance(wrong_test[i])
            test_wrong_dis.append(dis)
        test_wrong_dis = np.array(test_wrong_dis)
        print("############# to label %d ##############"%j)
        print(test_correct_dis)
        print(test_correct_dis.max())
        print(test_correct_dis.mean())
        print(test_wrong_dis)
        print(test_wrong_dis.max())
        print(test_wrong_dis.mean())
test_m_dist(3)