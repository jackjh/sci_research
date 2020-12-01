import numpy as np
import sys
import os

# 对向量进行(0, 1)归一化处理
def max_min_normalization(x, max_, min_):
    x = (x - min_) / (max_ - min_)
    return x

# 对向量进行标准化处理
def z_score_normalization(x, mean, sigma):
    x = (x - mean) / sigma
    return x

def lbp_hist_normalization(data_path):
    # info = np.loadtxt(data_path + "lbphistdr1.txt")
    info = np.loadtxt(data_path + "lbphist1.txt")
    print(info)
    # maxVal = np.max(info)
    # minVal = np.min(info)
    # normal_hist = max_min_normalization(info, maxVal, minVal)
    mean = np.average(info)
    sigma = np.std(info)
    print(mean)
    print(sigma)
    # normal_hist = z_score_normalization(info, mean, sigma)
    # np.savetxt("face_detect/train_normal/tmp_lbphistnorm1.txt", normal_hist, fmt='%.4f')

def lbp_hist_bat_normal(img_path, save_path, index):
    i = 1
    path = save_path + str(index) + "/"
    if not os.path.exists(path):
        os.mkdir(path)

    while i < 10:
        file_path = img_path + "lbphist" + str(i) + ".txt"
        info = np.loadtxt(file_path)
        mean = np.average(info)
        sigma = np.std(info)
        normal_hist = z_score_normalization(info, mean, sigma)
        save_file = path + "/lbphistnorm" + str(i) + ".txt"
        np.savetxt(save_file, normal_hist, fmt='%.4f')
        i += 1

def all_imp_normal(data_path, save_path, nums):
    index = 1
    while index < nums:
        img_path = data_path + str(index) + "/"
        lbp_hist_bat_normal(img_path, save_path, index)
        index += 1

# 降维后的向量中出现负数，该函数是消除负数
def lbp_hist_avoidNeg(data_path):
    info = np.loadtxt(data_path + "lbphistdr3.txt")
    print(info)
    # print(np.min(info))
    info = info + abs(np.min(info))
    print(info)

# train_path = "face_detect/train_hist/"
train_path = "face_detect/train_cut/"
train_save = "face_detect/train_normal/"
test_path = "face_detect/test_cut/"
test_save = "face_detect/test_normal/"

# all_imp_normal(train_path, train_save, 801)
all_imp_normal(test_path, test_save, 171)
# lbp_hist_avoidNeg(train_path)
# lbp_hist_bat_normal(train_path)

# res = z_score_normalization(new_lbp_hist, 0.0982, -0.0940)
# np.savetxt("face_detect/train1/normalization.txt", res, fmt='%.4f')