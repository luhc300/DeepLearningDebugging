from keras.datasets import mnist
from src.cnn_profiler import CNNProfiler
from src.configs.network_configs.mnist.network_config_6 import NETWORK_STRUCTURE, NETWORK_ANCHOR, NETWORK_PATH, INIT, LEARNING_RATE, DISTRIBUTION_PATH
from src.distribution import Distribution
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
def data_prepare(label):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32")
    X_train /= 255
    X_test = X_test.astype("float32")
    X_test /= 255
    X_train_remove = X_train[y_train!=label]
    y_train_remove = y_train[y_train != label]
    X_train_add = X_train[y_train == label][:10]
    X_train_remove = np.concatenate((X_train_remove, X_train_add), axis=0)
    y_train_add = y_train[y_train == label][:10]
    y_train_remove = np.concatenate((y_train_remove, y_train_add), axis=0)
    print(X_train_remove.shape)
    y_train_remove[(y_train_remove==1)|(y_train_remove==3)|(y_train_remove==5)|(y_train_remove==7)|(y_train_remove==9)] = 1
    y_train_remove[(y_train_remove==0)|(y_train_remove==2)|(y_train_remove==4)|(y_train_remove==6)|(y_train_remove==8)] = 0
    y_test[(y_test==1)|(y_test==3)|(y_test==5)|(y_test==7)|(y_test==9)] = 1
    y_test[(y_test==0)|(y_test==2)|(y_test==4)|(y_test==6)|(y_test==8)] = 0
    return X_train_remove, y_train_remove, X_test, y_test

def data_prepare_new(select_label, new_label):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32")
    X_train /= 255
    X_test = X_test.astype("float32")
    X_test /= 255
    X_train_new = []
    y_train_new = []
    for i in range(len(y_train)):
        if y_train[i] in select_label:
            X_train_new.append(X_train[i])
            y_train_new.append(y_train[i])
    X_train_new = np.array(X_train_new)
    y_train_new = np.array(y_train_new)
    X_test_new = []
    y_test_new = []
    extra_count = 0
    extra_label = 2
    for i in range(len(y_test)):
        if (y_test[i] in select_label):
            X_test_new.append(X_test[i])
            y_test_new.append(y_test[i])
        if y_test[i] in new_label:
            X_test_new.append(X_test[i])
            y_test_new.append(extra_label)
            extra_count += 1
    print(extra_count)
    X_test_new = np.array(X_test_new)
    y_test_new = np.array(y_test_new)
    return X_train_new, y_train_new, X_test_new, y_test_new

X_train, y_train, X_test, y_test = data_prepare_new([0, 1], [2, 3])
(X_train_ori, y_train_ori), (X_test_ori, y_test_ori) = mnist.load_data()
input_dim = [None, 28, 28, 1]
output_dim = [None, 3]
cnn_profiler = CNNProfiler(NETWORK_STRUCTURE, network_anchor=NETWORK_ANCHOR, network_path=NETWORK_PATH, init=INIT, lr=LEARNING_RATE)
total_label = 3
distribution_path = DISTRIBUTION_PATH
def train():
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    values = y
    n_values = total_label
    y = np.eye(n_values)[values]
    cnn_profiler.train(input_dim, output_dim, x, y, iter=1000)
def test():
    x = X_test.astype("float32")[y_test_ori]
    y = y_test.reshape(-1)[y_test_ori]
    values = y
    n_values = total_label
    y = np.eye(n_values)[values]
    acc = 0
    for i in range(0, 8001, 2000):
        acc += cnn_profiler.test(input_dim, output_dim, x[i:i+2000], y[i:i+2000])
    acc /= 5
    print(acc)
def test_with_filter(label, anchor, filter):
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label][:10000]
    y = y[y == label][:10000]
    values = y
    n_values = total_label
    y = np.eye(n_values)[values]
    correct = cnn_profiler.get_correct_mid(input_dim, output_dim, x, y, anchor=anchor,filter=filter)
    distribute_correct = Distribution(correct)
    own_dis = []
    for i in range(correct.shape[0]):
        dis = distribute_correct.mahanobis_distance(correct[i])
        own_dis.append(dis)
    own_dis = np.array(own_dis)
    # print(own_dis)
    print("############## with filter %d #############" % filter)
    print(own_dis.min())
    print(own_dis.max())
    print(own_dis.mean())
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    y_ori = y_test_ori
    x = x[y_ori== 4][:5000]
    y = y[y_ori== 4][:5000]
    wrong_test, pic = cnn_profiler.get_mid_by_label(input_dim, output_dim,  x, y, label, anchor=anchor,filter=filter)
    # logits, pic = cnn_profiler.get_mid_by_label(input_dim, output_dim,  x, y, label, anchor=-1)
    # pic = pic.reshape(28,28)
    # plt.imshow(pic)
    # plt.show()
    # print(logits)
    test_wrong_dis = []
    for i in range(wrong_test.shape[0]):
        dis = distribute_correct.mahanobis_distance(wrong_test[i])
        test_wrong_dis.append(dis)
    test_wrong_dis = np.array(test_wrong_dis)
    # print(test_correct_dis)
    # print(test_wrong_dis)
    print(test_wrong_dis.min())
    print(test_wrong_dis.max())
    print(test_wrong_dis.mean())
    return test_wrong_dis.mean() / own_dis.mean()
def get_distribution(label, anchor, filter):
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label][:10000]
    y = y[y == label][:10000]
    values = y
    n_values = total_label
    y = np.eye(n_values)[values]
    correct = cnn_profiler.get_correct_mid(input_dim, output_dim, x, y, anchor=anchor, filter=filter)
    # correct = cnn_profiler.get_mid(input_dim, output_dim, x, anchor=anchor, filter=filter)
    distribute_correct = Distribution(correct)
    own_dis = distribute_correct.cal_self_distance()
    # print(own_dis)
    print("######### Distribution for label %d #########" % label)
    print(own_dis.min())
    print(own_dis.max())
    print(own_dis.mean())
    return distribute_correct

def calc_distribution(anchor, filter, distribution_path):
    # if os.path.exists(distribution_path):
    #     with open(distribution_path, "rb") as distribution:
    #         distribution_list = pickle.load(distribution)
    # else:
    distribution_list = []
    for i in range(total_label):
        distribution_list.append(get_distribution(i, anchor, filter))
    with open(distribution_path, "wb") as distribution:
        pickle.dump(distribution_list, distribution)
    return distribution_list

def test_m_dist_by_label(label):
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    y_ori = y_test_ori
    x = x[y_ori == label]
    y = y[y_ori == label]
    values = y
    n_values = total_label
    y = np.eye(n_values)[values]
    anchor = 1
    filter = 28
    mid = cnn_profiler.get_mid(input_dim, output_dim, x, anchor=anchor, filter=filter)
    distribution_list = calc_distribution(anchor, filter, distribution_path)
    for j in range(total_label):
        distribute_correct = distribution_list[j]
        test_correct_dis = []
        for i in range(mid.shape[0]):
            dis = distribute_correct.mahanobis_distance(mid[i])
            test_correct_dis.append(dis)
        test_correct_dis = np.array(test_correct_dis)
        print("############# to label %d ##############"%j)
        # print(test_correct_dis)
        print(test_correct_dis.max())
        print(test_correct_dis.mean())


def test_m_dist_by_correctness(label):
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    y_ori = y_test_ori
    x = x[y_ori == label]
    y = y[y_ori == label]
    values = y
    n_values = total_label
    y = np.eye(n_values)[values]
    anchor = -2
    filter = None
    correct_test = cnn_profiler.get_correct_mid(input_dim, output_dim, x, y, anchor=anchor, filter=filter)
    wrong_test = cnn_profiler.get_correct_mid(input_dim, output_dim, x, y, wrong=True, anchor=anchor, filter=filter)
    distribution_list = calc_distribution(anchor, filter, distribution_path)
    for j in range(total_label):
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
        # print(test_correct_dis)
        print(test_correct_dis.max())
        print(test_correct_dis.mean())
        # print(test_wrong_dis)
        print(test_wrong_dis.max())
        print(test_wrong_dis.mean())
def test_wrong_filter():
    x = X_test.astype("float32")
    y = y_test.reshape(-1)
    x = x[5000:10000]
    y = y[5000:10000]
    anchor = -2
    filter = None
    mid = cnn_profiler.get_mid(input_dim, output_dim, x, anchor=anchor, filter=filter)
    result = cnn_profiler.test(input_dim, output_dim, x)
    distribution_list = calc_distribution(anchor, filter, distribution_path)
    range_filter(distribution_list, mid, result, y, 10)


def min_filter(distribution_list, mid, result, y, threshold):
    correct = 0
    total = 0
    correct_dropped = 0
    total_dropped = 0
    # mean_list = []
    # for j in range(total_label):
    #     data = distribution_list[j].data
    #     sum = 0
    #     for i in range(data.shape[0]):
    #         sum += distribution_list[j].mahanobis_distance(data[i])
    #     sum /= data.shape[0]
    #     mean_list.append(sum)
    # mean_list = np.array(mean_list)
    # print(mean_list)
    for i in range(y.shape[0]):
        dis = []
        for j in range(total_label):
            dis.append(distribution_list[j].mahanobis_distance(mid[i]))
        dis = np.array(dis)
        dis_validataion = np.array(dis)
        dis_validataion.sort()
        if (dis_validataion[1]/ dis_validataion[0] < threshold):
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

def range_filter(distribution_list, mid, result, y, threshold):
    correct = 0
    total = 0
    correct_dropped = 0
    total_dropped = 0
    for i in range(y.shape[0]):
        dis = []
        dis = distribution_list[result[i]].mahanobis_distance(mid[i])
        max = distribution_list[result[i]].self_distance.max()
        if (dis<max):
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
# max_mean = 0
# for i in range(32):
#     result = test_with_filter(1, 4, i)
#     if result > max_mean:
#         max_mean = result
# print(max_mean)

# train()
# test()
# test_m_dist_by_correctness(4)
test()
