#coding=utf8
from __future__ import division
from splitter.splitter import split_from_url, Splitter
from splitter.pic_classifier import classify_pic
import cv2
import os


root_path = '../data/test_10/'
target_path = '../data/test_10'
if not os.path.exists(target_path):
    os.mkdir(target_path)
img_files = os.listdir(root_path)

# 测试行切割
# 初始化模型，第一个参数为行切割模型路径，第二个参数为字符切割路径参数
sp = Splitter('splitter/models/model', 'splitter/models/char_model3')

# 切割
for img_file in img_files:
    if not img_file == '10.jpg':
        continue
    pic_list = sp.split_img(root_path + img_file)
    # 判断是否无法进行倾斜矫正，如果无法倾斜矫正会返回false
    if isinstance(pic_list, bool):
        print('wrong_pic%s' % root_path)
    # 保存pic_list

    if not os.path.exists(os.path.join(target_path, img_file.split('.')[0])):
        os.mkdir(os.path.join(target_path, img_file.split('.')[0]))
    for i, line in enumerate(pic_list):
        for j, pic in enumerate(line):
            for k, piece in enumerate(sp.split_char(pic)):
                split_pic = os.path.join(target_path, img_file.split('.')[0], '%d_%d_%d.png' % (i, j, k))
                cv2.imwrite(split_pic, piece)
assert 1==0
# 字符切割
mat = [[1, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 0], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [0, 0, 1, 1, 1], [1, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1]]
split2 = classify_pic(pic_list, mat, sp.split_char)
