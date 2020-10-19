import numpy as np
from sklearn.decomposition import PCA

# read_info = open("dataset/hist_train/lbp_hist_1.txt", "r").read()
info = np.loadtxt('dataset/hist_train/lbp_hist_1.txt')
lbp_hist = info.reshape(1, -1)
lbp_hist = lbp_hist.T

'''
info2 = np.loadtxt('dataset/hist_train/lbp_hist_2.txt')
lbp_hist2 = info2.reshape(1, -1)
lbp_hist2 = lbp_hist2.T
lbp_hist = np.hstack((lbp_hist1, lbp_hist2))

print(lbp_hist)
'''

i = 2
while i < 801:
    path = "dataset/hist_train/lbp_hist_" + str(i) + ".txt"
    load_info = np.loadtxt(path)
    lbp_hist_tmp = load_info.reshape(1, -1)
    lbp_hist_tmp = lbp_hist_tmp.T
    lbp_hist = np.hstack((lbp_hist, lbp_hist_tmp))
    i += 1

# print(lbp_hist)

pca = PCA(n_components=18)
pca.fit(lbp_hist)
new_lbp_hist = pca.fit_transform(lbp_hist)
print(pca.explained_variance_ratio_)
np.savetxt("dataset/pca_train/pca_lbp_hist.txt", new_lbp_hist, fmt='%.4f')
