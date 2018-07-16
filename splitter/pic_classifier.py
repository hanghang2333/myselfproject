#coding=utf8
from __future__ import division
import numpy as np
import cv2
from collections import Counter
import copy


def classify_pic(pic_list, class_mat, split_func):
    """
    Classify different pic from a pic list using pic location.
    """

    # firstly validate the class matrix
    new_class_mat = []
    wrong_lines = []
    for i, line in enumerate(class_mat):
        new_line = copy.deepcopy(line)
        # new_line = line.copy()

        # if there are less than 2 elements, we determine it a useless line
        if len(line) < 2:
            wrong_lines.append(i)
            continue

        # validate whether there are not 0 in a line
        if 0 not in line:
            if len(pic_list[i][0][0]) > len(pic_list[i][1][0]):
                new_line[0] = 0
            else:
                new_line[1] = 0

        # validate whether there are more than a 1 in a line
        #目前是取比较靠前的，改为比较长的
        if Counter(line)[0]>1:
            indx = []
            for k in range(len(line)):
                if line[k] is 0:
                    indx.append(k)
            idxlen = [len(pic_list[i][j][0]) for j in indx]
            maxlen = max(idxlen)
            idxmax = idxlen.index(maxlen)
            for idx, j in enumerate(line):
                if j == 0 and idx==indx[idxmax]:
                        pass
                elif j ==0:
                    new_line[idx] = 1

        '''
        if Counter(line)[0] > 1:
            flag = False
            for idx, j in enumerate(line):
                if not flag:
                    if j == 0:
                        flag = True
                else:
                    if j == 0:
                        new_line[idx] = 1
        '''
        # add new validated line to new class matrix
        new_class_mat.append(new_line)

    # delete the wrong line in pic_list
    wrong_lines.reverse()
    for wrong_line_idx in wrong_lines:
        del pic_list[wrong_line_idx]

    # classfy the pic and split char pictures
    new_pic_list = []
    #new_pic_list2 = []
    for i, line in enumerate(pic_list):
        new_pic_line = []
        #new_pic_line2 = []
        start = False
        for j, pic in enumerate(line):
            if start:
                if new_class_mat[i][j] == 1:
                    new_pic_line.append(split_func(pic))
                    #new_pic_line2.append(pic)
            else:
                if new_class_mat[i][j] == 0:
                    start = True
                    new_pic_line.append(pic)
                    #new_pic_line2.append(pic)

        new_pic_list.append(new_pic_line)
        #new_pic_list2.append(new_pic_line2)

    new_pic_list = np.array(new_pic_list)
    #new_pic_list2 = np.asarray(new_pic_list2)
    return new_pic_list#,new_pic_list2


if __name__ == '__main__':
    # len_list = [5, 7, 7, 6, 6, 6, 6, 7, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 3, 4]
    # mat = []
    # for i in len_list:
    #     mat.append([0 if j == 2 else 1 for j in range(i)])
    # print(mat)
    mat = [[1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 1, 0], [1, 1, 0, 1]]
    file_path = 'report/3dsf.jpeg'
