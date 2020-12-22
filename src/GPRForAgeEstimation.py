import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor


train_path = "face_detect/train_norm_hist/"
train_x_info = np.loadtxt(train_path + "lbphistdr1.txt")
train_x_data = train_x_info.reshape(1, -1)

test_path = "face_detect/test_norm_hist/"
test_x_info = np.loadtxt(test_path + "lbphistdr1.txt")
test_x_data = test_x_info.reshape(1, -1)

i = 2
while i < 801:
    file = train_path + "lbphistdr" + str(i) + ".txt";
    tmpInfo = np.loadtxt(file)
    tmpData = tmpInfo.reshape(1, -1)
    train_x_data = np.vstack((train_x_data, tmpData))
    i += 1

i = 2
while i < 171:
    file = test_path + "lbphistdr" + str(i) + ".txt";
    tmpInfo = np.loadtxt(file)
    tmpData = tmpInfo.reshape(1, -1)
    test_x_data = np.vstack((test_x_data, tmpData))
    i += 1

train_y_info = np.loadtxt("face_detect/ageYtrain.txt")
#train_y_data = train_y_info.reshape(1, -1)

arr = []
train_y_data = [[train_y_info[0]]]
for i in range(1, 800):
    tmp_arr = [[train_y_info[i]]]
    train_y_data = np.r_[train_y_data, tmp_arr]

#print(len(train_y_data))
#print(train_y_data)

kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
gpreg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

gpreg.fit(train_x_data, train_y_data)
y_pred, err = gpreg.predict(test_x_data, return_std=True)

print(y_pred)
print(err)
