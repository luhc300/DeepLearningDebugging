from keras.datasets import cifar10
from src.cnn_profiler import CNNProfiler
from src.configs.network_configs.cifar.network_config_3 import NETWORK_STRUCTURE, NETWORK_ANCHOR, NETWORK_PATH, INIT, LEARNING_RATE, DISTRIBUTION_PATH
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from src.distribution import Distribution
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32")
X_train /= 255
X_test = X_test.astype("float32")
X_test /= 255
distribution_path = DISTRIBUTION_PATH
ml_path = "profile/cifar/ml_filter_model.pkl"
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
    own_dis = []
    for i in range(correct.shape[0]):
        dis = distribute_correct.mahanobis_distance(correct[i])
        own_dis.append(dis)
    own_dis = np.array(own_dis)
    # print(own_dis)
    print("######### Distribution for label %d #########"%label)
    print(own_dis.min())
    print(own_dis.max())
    print(own_dis.mean())
    return distribute_correct
def calc_distribution(anchor, filter, distribution_path):
    if os.path.exists(distribution_path):
        with open(distribution_path, "rb") as distribution:
            distribution_list = pickle.load(distribution)
    else:
        distribution_list = []
        for i in range(10):
            distribution_list.append(get_distribution(i, anchor, filter))
        with open(distribution_path, "wb") as distribution:
            pickle.dump(distribution_list, distribution)
    return distribution_list
def test_m_dist(label):
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    x = x[y == label]
    y = y[y == label]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    anchor = -2
    filter = None
    correct_test = cnn_profiler.get_correct_mid([None, 32, 32, 3], [None, 10], x, y, anchor=anchor, filter=filter)[5:6]
    wrong_test = cnn_profiler.get_correct_mid([None, 32, 32, 3], [None, 10], x, y, wrong=True, anchor=anchor, filter=filter)[5:6]
    distribution_list = calc_distribution(anchor, filter, distribution_path)
    for j in range(10):
        distribute_correct = distribution_list[j]
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
        print("############# to label %d ##############"%j)
        print(test_correct_dis)
        print(test_correct_dis.max())
        print(test_correct_dis.mean())
        print(test_wrong_dis)
        print(test_wrong_dis.max())
        print(test_wrong_dis.mean())
def test_wrong_filter():
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    x = x[5000:10000]
    y = y[5000:10000]
    anchor = -2
    filter = None
    mid = cnn_profiler.get_mid([None, 32, 32, 3], [None, 10], x, anchor=anchor, filter=filter)
    result = cnn_profiler.test([None, 32, 32, 3], [None, 10], x)
    distribution_list = calc_distribution(anchor, filter, distribution_path)
    ml_filter(distribution_list, mid, result, y, 0.9, train=False)

def min_filter(distribution_list, mid, result, y, threshold):
    correct = 0
    total = 0
    correct_dropped = 0
    total_dropped = 0
    for i in range(y.shape[0]):
        dis = []
        for j in range(10):
            dis.append(distribution_list[j].mahanobis_distance(mid[i]))
        dis = np.array(dis)
        dis_validate = np.array(dis)
        dis_validate.sort()
        min_label = dis.argmin()
        if (dis_validate[1] - dis_validate[0]>5):
            total += 1
            if result[i] == y[i]:
                correct += 1
        else:
            total_dropped += 1
            if result[i] == y[i]:
                correct_dropped += 1
    print(correct, total)
    print(correct / total)
    print(correct_dropped, total_dropped)
    print(correct_dropped / total_dropped)

def boundary_filter(distribution_list, mid, result, y, threshold):
    correct = 0
    total = 0
    correct_dropped = 0
    total_dropped = 0
    for i in range(y.shape[0]):
        max_pred = mid[i].max()
        if max_pred >= threshold:
            total += 1
            if result[i] == y[i]:
                correct += 1
        else:
            total_dropped += 1
            if result[i] == y[i]:
                correct_dropped += 1
    print(correct, total)
    print(correct / total)
    print(correct_dropped, total_dropped)
    print(correct_dropped / total_dropped)
def ml_filter(distribution_list, mid, result, y, threshold, train=False):
    total_dis = []
    for i in range(y.shape[0]):
        dis = []
        for j in range(10):
            dis.append(distribution_list[j].mahanobis_distance(mid[i]))
        dis = np.array(dis)
        dis.sort()
        # dis = (dis - dis.min()) / (dis.max() - dis.min())
        for j in range(9):
            dis[j] = dis[j+1] - dis[0]
        dis = dis[:9]
        total_dis.append(dis)
    total_dis = np.array(total_dis)
    print(total_dis)
    correct_result = result == y
    correct_result = correct_result.astype("int")
    print(correct_result)
    if train:
        rf =RandomForestClassifier()
        rf.fit(total_dis, correct_result)
        with open(ml_path, "wb") as file:
            pickle.dump(rf, file)
    else:
        with open(ml_path, "rb") as file:
            rf = pickle.load(file)
        pred = rf.predict(total_dis)
        correct = 0
        total = 0
        correct_dropped = 0
        total_dropped = 0
        for i in range(y.shape[0]):
            if pred[i] == 1:
                total += 1
                if result[i] == y[i]:
                    correct += 1
            else:
                total_dropped += 1
                if result[i] == y[i]:
                    correct_dropped += 1
        print(correct, total)
        print(correct / total)
        print(correct_dropped, total_dropped)
        print(correct_dropped / total_dropped)

test_wrong_filter()