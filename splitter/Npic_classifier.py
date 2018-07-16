#coding=utf8
from __future__ import division
import numpy as np
import cv2
from collections import Counter
from .char_splitter import char_split


def classify_pic(pic_list, class_mat,split_func):
    """
    Classify different pic from a pic list using pic location.
    """
    new_pic_list = []
    for i, line in enumerate(pic_list):
        new_pic_line = []
        for j, pic in enumerate(line):
            if class_mat[i][j] == 1:
                new_pic_line.append(split_func(pic))
            else:
                new_pic_line.append(pic)

        new_pic_list.append(new_pic_line)

    new_pic_list = np.array(new_pic_list)
    return new_pic_list


if __name__ == '__main__':
    # len_list = [5, 7, 7, 6, 6, 6, 6, 7, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 3, 4]
    # mat = []
    # for i in len_list:
    #     mat.append([0 if j == 2 else 1 for j in range(i)])
    # print(mat)
    mat = [[1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0], [1, 1, 0, 1]]
    file_path = 'report/3dsf.jpeg'
