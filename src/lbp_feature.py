from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io, data_dir, filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

radius = 1
n_point = radius * 8

# image = cv2.imread('face_detect/2_0.bmp')
# img_del = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.subplot(111)
# plt.imshow(img_del)
# plt.show()

# train_read = "face_detect/train/"
# test_read = "face_detect/test/"
# cell_read = "face_detect/train1/img_save/"

log_file = "log/lbp_log.txt"

# 计算每张图像分割后的9个小区域的lbp直方图
def compute_hist_per_img(img_path):
    img_list = os.listdir(img_path)
    img_list.sort(key=lambda x:int(x.split('.')[0]))
    index = 0
    for elem in img_list:
        file = img_path + elem
        image = cv2.imread(file)
        # 灰度图转换
        img_del = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # plt.subplot(111)
        # plt.imshow(img_del, plt.cm.gray)
        # plt.show()

        # LBP处理
        lbp = local_binary_pattern(img_del, n_point, radius)
        # plt.subplot(111)
        # plt.imshow(lbp, plt.cm.gray)
        # plt.show()
        # print(lbp)
        max_bins = int(lbp.max() + 1)

        lbp_hist, lbp_bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
        index = index + 1
        filename_hist = img_path + "lbphist" + str(index) + ".txt"
        np.savetxt(filename_hist, lbp_hist, fmt='%.4f')
        # np.savetxt(filename_bins, lbp_bins, fmt='%.4f')

def compute_hist(data_path, num):
    index = 1
    while index < num:
        img_path = data_path + str(index) + "/"
        fp = open(log_file, 'a+')
        fp.write(img_path + '\n')
        compute_hist_per_img(img_path)
        index += 1

def compute_equa_hist(data_path, save_path, num):
    index = 1
    while index < num:
        file = data_path + str(index) + ".JPG"
        image = cv2.imread(file)
        # 灰度图转换
        img_del = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # LBP处理
        lbp = local_binary_pattern(img_del, n_point, radius)
        max_bins = int(lbp.max() + 1)
        lbp_hist, lbp_bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

        hist_file = save_path + "hist" + str(index) + ".txt"
        np.savetxt(hist_file, lbp_hist, fmt='%.4f')
        index = index + 1

def computeImgEquaHist(img):
    image = cv2.imread(img)
    # 灰度图转换
    img_del = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # LBP处理
    lbp = local_binary_pattern(img_del, n_point, radius)
    max_bins = int(lbp.max() + 1)
    lbp_hist, lbp_bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    np.set_printoptions(formatter={'float':'{:.4f}'.format})
    resArr = np.array(lbp_bins)

def getEquaHistMatrix(data_path, num):
    index = 1
    while index < num:
        file = data_path + str(index) + ".JPG"


train_path = "face_detect/train_cut_4/"
test_path = "face_detect/test_cut_4/"

train_path_equa = "face_detect/train_cut_equa/"
test_path_equa = "face_detect/test_cut_equa/"

#compute_hist(train_path, 801)
#compute_hist(test_path, 171)

#compute_hist(train_path_equa, 801)
#compute_hist(test_path_equa, 171)

img_file = "face_detect/train_equa/11.JPG"
image = cv2.imread(img_file)
# 灰度图转换
img_del = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(111)
plt.imshow(img_del, plt.cm.gray)
plt.show()

# LBP处理
lbp = local_binary_pattern(img_del, n_point, radius)
plt.subplot(111)
plt.imshow(lbp, plt.cm.gray)
plt.show()
print(lbp)
max_bins = int(lbp.max() + 1)
lbp_hist, lbp_bins = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
filename = "img_equa_hist.txt"
np.savetxt(filename, lbp_hist, fmt='%.4f')


'''
def load_image(img_path):
    image = Image.open(img_path)
    img = np.array(image)
    


def lbp_texture_detect():
    im_hist = np.zeros((200, 256))

    return im_hist
# end of func lbp_texture_detect
'''