import numpy as np
import sys

# 对向量进行归一化处理
def z_score_normalization(x, max_, min_):
    x = (x - min_) / (max_ - min_)
    return x

def lbp_hist_normalization(data_path):
    info = np.loadtxt(data_path + "lbphistdr1.txt")
    print(info)

train_path = "face_detect/train_hist/"
lbp_hist_normalization(train_path)

# res = z_score_normalization(new_lbp_hist, 0.0982, -0.0940)
# np.savetxt("face_detect/train1/normalization.txt", res, fmt='%.4f')