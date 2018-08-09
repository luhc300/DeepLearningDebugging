import numpy as np


def brightness(data_x, data_y, times=1):
    result_x = None
    result_y = None
    ori_shape = data_x.shape
    for i in range(times):
        pert = np.random.randint(-10, 10, data_x.shape[0])*10
        print(pert)
        pert = pert.reshape(-1, 1)
        temp = data_x.reshape(data_x.shape[0], -1) + pert
        temp[temp > 255] = 255
        temp[temp < 0] = 0
        temp = temp.reshape(ori_shape)
        if result_x is None:
            result_x = temp
            result_y = data_y
        else:
            result_x = np.concatenate((result_x, temp), axis=0)
            result_y = np.concatenate((result_y, data_y), axis=0)
    return result_x, result_y