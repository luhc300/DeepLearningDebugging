import numpy as np


class Distribution:
    def __init__(self, data):
        self.data = data
        self.center = np.mean(data, axis=0)
        self.cov = np.cov(data.T)
        self.cov_inv = np.linalg.pinv(self.cov)
    def get_center(self):
        return self.center

    def get_covariance(self):
        return self.cov

    def mahanobis_distance(self, x):
        return np.squeeze(np.sqrt(np.dot(np.dot((x - self.center), self.cov_inv), (x - self.center).T)))


