import numpy as np
from PIL import Image
import cv2
import os


# 将图片分割成 3 * 3 个小区域
def image_cut(img):
    # image = Image.open("face_detect/train1/1_3.JPG")
    image = Image.open(img)
    width, height = image.size
    new_img_len = width
    if width > height:
        new_img_len = height

    new_img = Image.new(image.mode, (new_img_len, new_img_len), color='white')
    if width > height:
        new_img.paste(image, (0, int((new_img_len - height) / 2)))
    else:
        new_img.paste(image, (int((new_img_len - width) / 2), 0))

    new_width, new_height = new_img.size
    item_width = int(new_width / 3)
    box_list = []
    for i in range(0, 3):
        for j in range(0, 3):
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)

    img_list = [new_img.crop(box) for box in box_list]

    return img_list


'''
    index = 1
    for img in img_list:
        img.save('face_detect/train1/img_save/' + str(index) + '.JPG')
        index += 1
'''


train_read = "face_detect/test_tmp/"
train_cut_save = "face_detect/test_cut/"
index = 0
path_list = os.listdir(train_read)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for idx in path_list:
    img = train_read + idx
    img_list = image_cut(img)
    index += 1
    save_path = train_cut_save + str(index) + "/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cut_idx = 1
    for elem in img_list:
        img_new = elem.convert('RGB')
        img_new.save(save_path + str(cut_idx) + '.JPG')
        cut_idx += 1
