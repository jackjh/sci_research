import os

path = "face_detect/test_tmp/"
chpath = "E:/Research/code/PycharmProjects/sci_research/face_detect/test_tmp/"
file_list = os.listdir(path)
for file in file_list:
    pos = file.find("_")
    new_name = file[pos+1:]
    os.chdir(chpath)
    os.rename(file, new_name)
