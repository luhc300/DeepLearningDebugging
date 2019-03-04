import numpy as np


class Distribution:
    def __init__(self, data):
        self.data = data
        self.center = np.mean(data, axis=0)
        self.cov = np.cov(data.T)
        self.cov_inv = np.linalg.pinv(self.cov)
        self.self_distance = None
    def get_center(self):
        return self.center

    def get_covariance(self):
        return self.cov

    def mahanobis_distance(self, x):
        return np.squeeze(np.sqrt(np.dot(np.dot((x - self.center), self.cov_inv), (x - self.center).T)))

    def cal_self_distance(self):
        self.self_distance = []
        for i in range(self.data.shape[0]):
            self.self_distance.append(self.mahanobis_distance(self.data[i]))
        self.self_distance = np.array(self.self_distance)
        return self.self_distance


