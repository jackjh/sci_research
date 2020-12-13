import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

#kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
#reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

train_path = "face_detect/train_norm_hist/"
train_info = np.loadtxt(train_path + "lbphistdr1.txt")
train_data = train_info.reshape(1, -1)

test_path = "face_detect/test_norm_hist/"
test_info = np.loadtxt(test_path + "lbphistdr1.txt")
test_data = test_info.reshape(1, -1)

def combineXtrainSet(data_path, train_data):
    i = 2
    while i < 801:
        file = train_path + "lbphistdr" + str(i) + ".txt";
        tmpInfo = np.loadtxt(file)
        tmpData = tmpInfo.reshape(1, -1)
        train_data = np.vstack((train_data, tmpData))
        i += 1

def combineXtestSet(data_path, test_data):
    i = 2
    while i < 171:
        file = test_path + "lbphistdr" + str(i) + ".txt";
        tmpInfo = np.loadtxt(file)
        tmpData = tmpInfo.reshape(1, -1)
        test_data = np.vstack((test_data, tmpData))
        i += 1

'''

info1 = np.loadtxt(train_path + "lbphistdr1.txt")
data1 = info1.reshape(1, -1)
info2 = np.loadtxt(train_path + "lbphistdr2.txt")
data2 = info2.reshape(1, -1)
data = np.vstack((data1, data2))
print(data)

'''