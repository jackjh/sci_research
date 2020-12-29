import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

train_read = "face_detect/train_tmp/"
train_equa_save = "face_detect/train_equa/"

test_read = "face_detect/test_tmp/"
test_equa_save = "face_detect/test_equa/"

def img_hist_equa(img):
    image = cv.imread(img)
    im_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # 将图像转为灰度，减少计算的维度
    # cv.imshow('im_gray', im_gray)

    height = im_gray.shape[0]
    width = im_gray.shape[1]

    # plt.hist(im_gray.reshape(im_gray.size, 1))
    # plt.show()

    # 创建直方图
    n = np.zeros((256), dtype=np.float)
    p = np.zeros((256), dtype=np.float)
    c = np.zeros((256), dtype=np.float)

    # 遍历图像的每个像素,得到统计分布直方图
    for x in range(0, im_gray.shape[0]):
        for y in range(0, im_gray.shape[1]):
            n[im_gray[x][y]] += 1

    # 归一化
    for i in range(0, 256):
        p[i] = n[i] / float(im_gray.size)

    # 计算累积直方图
    c[0] = p[0]
    for i in range(1, 256):
        c[i] = c[i - 1] + p[i]

    des = np.zeros((height, width), dtype=np.uint8)
    for x in range(0, height):
        for y in range(0, width):
            des[x][y] = 255 * c[im_gray[x][y]]

    # cv.imshow('des', des)
    # plt.hist(des.reshape(des.size, 1))
    # plt.show()

    return des

def bat_equalization(data_path, save_path):
    img_list = os.listdir(data_path)
    img_list.sort(key=lambda x:int(x.split('.')[0]))
    index = 0
    for elem in img_list:
        img = data_path + elem
        res = img_hist_equa(img)
        index += 1
        tmp = Image.fromarray(res)
        img_file = tmp.convert('RGB')
        img_file.save(save_path + str(index) + '.JPG')

# 训练集直方图均衡化
# bat_equalization(train_read, train_equa_save)

# 测试集直方图均衡化
bat_equalization(test_read, test_equa_save)


