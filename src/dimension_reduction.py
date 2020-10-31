import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import sys

'''
# read_info = open("dataset/hist_train/lbp_hist_1.txt", "r").read()
# info = np.loadtxt('dataset/hist_train/lbp_hist_1.txt')
info = np.loadtxt('face_detect/train1/lbp_hist/lbp_hist_1.txt')
lbp_hist = info.reshape(1, -1)


i = 2
# while i < 801:
while i < 10:
    # path = "dataset/hist_train/lbp_hist_" + str(i) + ".txt"
    path = "face_detect/train1/lbp_hist/lbp_hist_" + str(i) + ".txt"
    load_info = np.loadtxt(path)
    lbp_hist_tmp = load_info.reshape(1, -1)
    lbp_hist = np.vstack((lbp_hist, lbp_hist_tmp))
    i += 1

print(lbp_hist)

pca = PCA(n_components=1)
pca.fit(lbp_hist)
new_lbp_hist = pca.fit_transform(lbp_hist)
print("各主成分占比：", pca.explained_variance_ratio_)
# np.savetxt("dataset/pca_train/pca_lbp_hist_1.txt", new_lbp_hist, fmt='%.4f')
np.savetxt("face_detect/train1/pca_lbp_hist_1.txt", new_lbp_hist, fmt='%.4f')
'''

log_file = "log/dim_red_log.txt"

# 给每张图像分割后的9个小区域连成直方图（9 x 256维）降维，将成 9 x 1 维
def dimRedPerImgHist(img_path, save_path, index):
    info = np.loadtxt(img_path + "lbphistnorm1.txt")
    lbp_hist = info.reshape(1, -1)
    i = 2
    while i < 10:
        tmp_path = img_path + "lbphistnorm" + str(i) + ".txt"
        tmp_info = np.loadtxt(tmp_path)
        tmp_hist = tmp_info.reshape(1, -1)
        lbp_hist = np.vstack((lbp_hist, tmp_hist))
        i += 1

    pca = PCA(n_components=1)
    pca.fit(lbp_hist)
    new_lbp_hist = pca.fit_transform(lbp_hist)
    save_file = save_path + "lbphistdr" + str(index) + ".txt"
    np.savetxt(save_file, new_lbp_hist, fmt='%.4f')

# 所有图像进行降维
def dimRedAllHist(data_path, save_path, nums):
    index = 1
    while index < nums:
        img_path = data_path + str(index) + "/"
        fp = open(log_file, 'a+')
        fp.write(img_path + '\n')
        dimRedPerImgHist(img_path, save_path, index)
        index += 1

train_path = "face_detect/train_cut/"
test_path = "face_detect/test_cut/"

train_save = "face_detect/train_hist/"
test_save = "face_detect/test_hist/"

data_path = "face_detect/train_normal/"
save_path = "face_detect/train_normal/"

# dimRedAllHist(train_path, train_save, 801)
# dimRedAllHist(test_path, test_save, 171)

dimRedPerImgHist(data_path, save_path, 1)
