import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import sys


# read_info = open("dataset/hist_train/lbp_hist_1.txt", "r").read()
# info = np.loadtxt('dataset/hist_train/lbp_hist_1.txt')
info = np.loadtxt('face_detect/train1/lbp_hist/lbp_hist_1.txt')
lbp_hist = info.reshape(1, -1)

'''
info2 = np.loadtxt('dataset/hist_train/lbp_hist_2.txt')
lbp_hist2 = info2.reshape(1, -1)
lbp_hist2 = lbp_hist2.T
lbp_hist = np.hstack((lbp_hist1, lbp_hist2))
print(lbp_hist)
'''

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


def z_score_normalization(x, max_, min_):
    x = (x - min_) / (max_ - min_)
    return x


res = z_score_normalization(new_lbp_hist, 0.0982, -0.0940)
np.savetxt("face_detect/train1/normalization.txt", res, fmt='%.4f')


'''
# 将图片分割成 3 * 3 个小区域
image = Image.open("face_detect/train1/1_3.JPG")
width, height = image.size
new_img_len = width
if width < height:
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

index = 1
for img in img_list:
    img.save('face_detect/train1/img_save/' + str(index) + '.JPG')
    index += 1
'''
