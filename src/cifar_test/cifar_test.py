from keras.datasets import cifar10
from src.cnn_profiler import CNNProfiler
from src.configs.network_configs.cifar.network_config_2 import NETWORK_STRUCTURE, NETWORK_ANCHOR, NETWORK_PATH, INIT, LEARNING_RATE
import numpy as np
import matplotlib.pyplot as plt
from src.distribution import Distribution
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
cnn_profiler = CNNProfiler(NETWORK_STRUCTURE, network_anchor=NETWORK_ANCHOR, network_path=NETWORK_PATH, init=INIT, lr=LEARNING_RATE)
################## Train ######################
def train():

    x = X_train.astype("float32")
    x /= 255
    print(X_train)
    y = y_train.reshape(-1)
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    cnn_profiler.train([None, 32, 32, 3], [None, 10], x, y, iter=6000)
################## Test ########################
def test():
    x = X_test.astype("float32")
    x /= 255
    y = y_test.reshape(-1)
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    cnn_profiler.test([None, 32, 32, 3], [None, 10], x[:1000], y[:1000])
################## Correct Vector ###################
def get_mid(label):
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label]
    mid = cnn_profiler.get_mid([None, 32, 32, 3], [None, 10], x / 255)
    return mid
def get_correct_mid(label):
    x = X_train.astype("float32")
    x /= 255
    y = y_train.reshape(-1)
    x = x[y == label]
    y = y[y == label]
    values = y
    n_values = 10
    y = np.eye(n_values)[values]
    print(y.shape)
    return cnn_profiler.get_correct_mid([None, 32, 32, 3], [None, 10], x, y)
################# Pertubation Test ####################
def pertubation_test(label, pertubation):
    def pertubate_img(img, pertubation):
        img_p = np.array(img)
        img_p += pertubation
        img_p[img_p>255]=255
        img_p[img_p<0]=0
        return img_p
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x_ori = x[y == label][1]
    print(x_ori.shape)
    x_ptb = pertubate_img(x_ori,pertubation)
    # x_ptb = x[y == (label+1)][0]
    plt.subplot(211)
    plt.imshow( x_ori/255)
    plt.subplot(212)
    plt.imshow( x_ptb/255)
    # plt.show()
    cnn_profiler.test([None, 32, 32, 3], [None, 10], x_ptb/255)
    ori_vec = cnn_profiler.get_mid([None, 32, 32, 3], [None, 10], x_ori / 255)
    ptb_vec = cnn_profiler.get_mid([None, 32, 32, 3], [None, 10], x_ptb/255)
    np.savetxt("profile/cifar/%s.txt" % str(label)+"_ori", ori_vec, fmt="%.4f")
    np.savetxt("profile/cifar/%s.txt" % str(label)+"_ptb", ptb_vec, fmt="%.4f")
    thres = 1e-2
    same = np.where(ori_vec-ptb_vec<=thres)
    diff = np.where(ori_vec-ptb_vec>=thres)
    print(same[1].shape, diff[1].shape)
    return diff[1]
# final_diff = None
# for i in range(-100,100,10):
#     print(i)
#     if i==0:
#         continue
#     diff = pertubation_test(6, i)
#     if final_diff is None:
#         final_diff = diff
#     else:
#         final_diff = np.intersect1d(final_diff, diff)
# print(final_diff.shape)
# print(final_diff)
################# Gaussian Check ####################
def gaussian_check(label):
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label]
    n, p = cnn_profiler.gaussian_check([None, 32, 32, 3], [None, 10], x)
    thres=1e-3
    p_normal = np.where(p>thres)[0]
    print(p_normal)
    return p_normal
################# Mahanobis Distance #################
def test_m_distance(label):
    x = X_train.astype("float32")
    y = y_train.reshape(-1)
    x = x[y == label]
    y = y[y == label]
    correct = get_correct_mid(label)
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
    correct_test = cnn_profiler.get_correct_mid([None, 32, 32, 3], [None, 10], xx, yy)
    wrong_test = cnn_profiler.get_correct_mid([None, 32, 32, 3], [None, 10], xx, yy, wrong=True)
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
train()





