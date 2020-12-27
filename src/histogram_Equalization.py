import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

train_read = "face_detect/train_tmp/"
train_equa_save = "face_detect/train_equa/"

test_read = "face_detect/test_tmp/"
test_equa_save = "face_detect/test_equa/"

img_file = train_read + "11.JPG"
img = cv.imread(img_file)

im_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)    #将图像转为灰度，减少计算的维度
cv.imshow('im_gray', im_gray)

#w = im_gray.shape[0]
#h = im_gray.shape[1]

print(im_gray)

p1 = plt.hist(im_gray.reshape(im_gray.size,1))
plt.show()

# 创建直方图
n = np.zeros((256), dtype=np.float)
p = np.zeros((256), dtype=np.float)
c = np.zeros((256), dtype=np.float)

#遍历图像的每个像素,得到统计分布直方图
for x in range(0, im_gray.shape[0]):
    for y in range(0, im_gray.shape[1]):
        n[im_gray[x][y]] += 1

print(n)