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

train_read = "face_detect/train/"
test_read = "face_detect/test/"
index = 0
for i in os.listdir(test_read):
    file = test_read + i
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
    hist_path = 'dataset/hist_test/'
    # filename_bins = 'lbp_bins.txt'
    index = index + 1
    filename_hist = hist_path + "lbp_hist_" + str(index) + ".txt"
    np.savetxt(filename_hist, lbp_hist, fmt='%.4f')
    # np.savetxt(filename_bins, lbp_bins, fmt='%.4f')


'''
def load_image(img_path):
    image = Image.open(img_path)
    img = np.array(image)
    


def lbp_texture_detect():
    im_hist = np.zeros((200, 256))

    return im_hist
# end of func lbp_texture_detect
'''