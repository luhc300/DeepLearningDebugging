from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from src.cnn_profiler import CNNProfiler
from src.configs.network_configs.mnist.network_config_1 import NETWORK_STRUCTURE, NETWORK_ANCHOR, NETWORK_PATH
from src.distribution import Distribution
import numpy as np
mnist = read_data_sets("data/MNIST_data/", one_hot=False)
mnist_not_onehot = read_data_sets("data/MNIST_data/", one_hot=False)
cnn_profiler = CNNProfiler(NETWORK_STRUCTURE, network_anchor=NETWORK_ANCHOR, network_path=NETWORK_PATH)
# x = mnist.train.images
# y = mnist.train.labels
# cnn_profiler.train([None, 28, 28, 1], [None, 10], x, y)
# x = mnist.test.images[:500]
# y = mnist.test.labels[:500]
# cnn_profiler.test([None, 28, 28, 1], [None, 10], x, y)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

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
    print(own_dis)
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
    print(test_correct_dis)
    print(test_correct_dis.max())
    print(test_correct_dis.mean())
    print(test_wrong_dis)
    print(test_wrong_dis.max())
    print(test_wrong_dis.mean())
    x_another = X_test.astype("float32")
    y_another = y_test.reshape(-1)
    x_another = x_another[y_another == label+1]
    y_another = y_another[y_another == label+1]
    mid = cnn_profiler.get_mid([None, 28, 28, 1], [None, 10], x_another)
    another_dis = []
    for i in range(mid.shape[0]):
        dis = distribute_correct.mahanobis_distance(mid[i])
        another_dis.append(dis)
    another_dis = np.array(another_dis)
    print(another_dis)
    print(another_dis.max())
    print(another_dis.mean())
test_m_distance(4)
