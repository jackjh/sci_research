import dlib
import numpy as np
import cv2
import os

# img_read = cv2.imread('pic/001A18.JPG')
path_read = "dataset/tmp/"
img_save = "face_detect/train1/"

# a = img_read.shape

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# faces = detector(img_read)
# print("人脸数：", len(faces))

index = 770
for path in os.listdir(path_read):
    img_path = path_read + path
    img_read = cv2.imread(img_path)
    a = img_read.shape
    index = index + 1
    faces = detector(img_read)

    for k, d in enumerate(faces):
        pos_start = tuple([d.left(), d.top()])
        height = d.bottom() - d.top()
        if a[1] >= d.right():
            pos_end = tuple([d.right(), d.bottom()])
            width = d.right() - d.left()
        else:
            pos_end = tuple([a[1], d.bottom()])
            width = a[1] - d.left()

        img_blank = np.zeros((height + 1, width + 1, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                img_blank[i][j] = img_read[d.top() + i][d.left() + j]

        # cv2.imshow("detect_faces", img_blank)
        # cv2.waitKey(0)
        cv2.imwrite(img_save + str(k + 1) + "_" + str(index) + ".JPG", img_blank)
