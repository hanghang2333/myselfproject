#coding=utf8
from __future__ import division
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .preprocess import preprocess
from .train import process_img_per3pixel
from .test import post_process_per3pixel, binary_and_resize
import random


def detect(img, raw_img):
    """
    输入为二值化处理后的图片地址，之后还需加入倾斜矫正。
    输出为一个二维的列表。
    列表的每个元素表示一行数据。
    每行数据由一系列方框构成，每个方框由一个列表表示，列表元素为[start_x, start_y, end_x, end_y]，四个元素代表左上和右下两个点的绝对坐标。
    :param img_path:
    :return:
    """
    height, width = img.shape
    reversed_img = np.array([[float(255 - i) for i in j] for j in img])
    threshold = 30 * 255
    col_threshold = 2 * 255
    row_shadow = np.sum(reversed_img, axis=1)

    top_line = []
    bottom_line = []
    flag = False
    for i, sha in enumerate(row_shadow):
        if i == 0:
            continue
        if (sum(row_shadow[i: i+1]) > threshold) and (sum(row_shadow[i-1: i]) < threshold) and not flag:
            top_line.append(max(0, i-5))
            flag = True
        elif (sum(row_shadow[i: i+2]) < 2 * threshold) and (sum(row_shadow[i-2: i]) > 2 * threshold) and flag:
            bottom_line.append(i + 2)
            flag = False
    else:
        if flag:
            bottom_line.append(height)
            flag = False

    axis_list = []

    left_margin = int(width / 200)
    right_margin = int(width / 60)

    for i in range(len(bottom_line)):
        # Exclude some wrong line
        if bottom_line[i] - top_line[i] < 10:
            continue

        axis_line = []
        flag = False
        this_row = reversed_img[top_line[i]: bottom_line[i]]
        # cv2.imwrite('../data/line_data/' + str(random.randint(1, 100000)) + '.png', raw_img[top_line[i]: bottom_line[i]])
        col_shadow = np.sum(this_row, axis=0)
        for j, sh in enumerate(col_shadow[left_margin: -right_margin]):
            if (sh > col_threshold) and (col_shadow[j-1] <= col_threshold) and not flag:
                axis = [j, top_line[i], bottom_line[i]]
                flag = True
            elif (sum(col_shadow[j: j+right_margin]) < col_threshold) and flag:
                if j - axis[0] > 5:
                    axis.insert(2, j)
                    axis_line.append(axis)
                flag = False
        else:
            if flag:
                axis.insert(2, width)
                axis_line.append(axis)
        axis_list.append(axis_line)

    # pic_list = [raw_img[rect[1]: rect[3], rect[0]: rect[2]] for line in axis_list for rect in line]
    pic_list = [[raw_img[rect[1]: rect[3], rect[0]: rect[2]] for rect in line] for line in axis_list]
    # for i in pic_list:
    #     cv2.imshow('1', i)
    #     cv2.waitKey()
    img = Image.fromarray(raw_img)
    draw = ImageDraw.Draw(img)
    for rects in axis_list:
        for rect in rects:
            draw.rectangle(rect)
    img = np.array(img)
    return img, pic_list


def detect_by_hough(img, raw_img):
    """
    has the same function with detect(img, raw_img).
    but this function use hough line detection to detect each line.
    """
    height, width = img.shape
    print(width)
    reversed_img = np.array([[float(255 - i) for i in j] for j in img])

    img = np.array([[255 - i for i in j] for j in img], dtype=np.uint8)
    img = cv2.dilate(img, np.uint8([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), iterations=10)
    img = np.array([[255 - i for i in j] for j in img], dtype=np.uint8)
    cv2.imwrite('test.png', img)
    col_threshold = 2 * 255
    tmp_img = img.copy()
    # Start line detect
    lines = cv2.HoughLinesP(img, 1, np.pi/180, int(width * 0.2), minLineLength=width * 0.4, maxLineGap=4)
    print(lines.shape)

    top_line = []
    bottom_line = []

    if isinstance(lines, np.ndarray):
        lines = np.reshape(lines, [-1, 4])
        for line in lines:
            cv2.line(tmp_img, (line[0], line[1]), (line[2], line[3]), 0, )
        cv2.imwrite('test.png', tmp_img)
        new_lines = []
        for line in lines:
            # determine whether the line is horizonized or not
            if abs(line[2] - line[0]) < abs(line[3] - line[1]):
                # ensure that the second point in the right place
                if line[1] > line[3]:
                    line[0], line[1], line[2], line[3] = line[2], line[3], line[0], line[1]
                new_line = list(line)
                new_line.append((line[0] + line[2]) / 2)
                new_lines.append(new_line)
        new_lines.sort(key=lambda x: x[4])
        # 逐行进行扫描
        last_line = [0, 0, 0, 0, -10]
        for line in new_lines:
            if line[4] - last_line[4] > 5:
                bottom_line.append(max(line[0], line[2]))
                top_line.append(min(last_line[0], last_line[2]))
            last_line = line
        if bottom_line[0] < 5:
            del bottom_line[0]
            del top_line[0]
    else:
        print(1)
        return detect(img, raw_img)


    axis_list = []

    left_margin = int(width / 200)
    right_margin = int(width / 60)

    for i in range(len(bottom_line)):
        # Exclude some wrong line
        if bottom_line[i] - top_line[i] < 10:
            continue

        axis_line = []
        flag = False
        this_row = reversed_img[top_line[i]: bottom_line[i]]
        col_shadow = np.sum(this_row, axis=0)
        for j, sh in enumerate(col_shadow[left_margin: -right_margin]):
            if (sh > col_threshold) and (col_shadow[j-1] <= col_threshold) and not flag:
                axis = [j, top_line[i], bottom_line[i]]
                flag = True
            elif (sum(col_shadow[j: j+right_margin]) < col_threshold) and flag:
                if j - axis[0] > 5:
                    axis.insert(2, j)
                    axis_line.append(axis)
                flag = False
        else:
            if flag:
                axis.insert(2, width)
                axis_line.append(axis)
        axis_list.append(axis_line)

    # pic_list = [raw_img[rect[1]: rect[3], rect[0]: rect[2]] for line in axis_list for rect in line]
    pic_list = [[raw_img[rect[1]: rect[3], rect[0]: rect[2]] for rect in line] for line in axis_list]
    # for i in pic_list:
    #     cv2.imshow('1', i)
    #     cv2.waitKey()
    img = Image.fromarray(raw_img)
    draw = ImageDraw.Draw(img)
    for rects in axis_list:
        for rect in rects:
            draw.rectangle(rect)
    img = np.array(img)
    return img, pic_list


def detect_by_network(img, raw_img, split_func, line_split_sess):
    """
    输入为二值化处理后的图片地址，之后还需加入倾斜矫正。
    输出为一个二维的列表。
    列表的每个元素表示一行数据。
    每行数据由一系列方框构成，每个方框由一个列表表示，列表元素为[start_x, start_y, end_x, end_y]，四个元素代表左上和右下两个点的绝对坐标。
    :param img_path:
    :return:
    """
    height, width = img.shape
    reversed_img = np.array([[float(255 - i) for i in j] for j in img])
    threshold = 30 * 255
    col_threshold = 2 * 255
    row_shadow = np.sum(reversed_img, axis=1)

    top_line = []
    bottom_line = []
    flag = False
    for i, sha in enumerate(row_shadow):
        if i == 0:
            continue
        if (sum(row_shadow[i: i+1]) > threshold) and (sum(row_shadow[i-1: i]) < threshold) and not flag:
            top_line.append(max(0, i-3))
            flag = True
        elif (sum(row_shadow[i: i+2]) < 2 * threshold) and (sum(row_shadow[i-2: i]) > 2 * threshold) and flag:
            bottom_line.append(i + 2)
            flag = False
    else:
        if flag:
            bottom_line.append(height)
            flag = False

    axis_list = []

    left_margin = int(width / 200)
    right_margin = int(width / 60)

    for i in range(len(bottom_line)):
        # Exclude some wrong line
        if bottom_line[i] - top_line[i] < 10:
            continue
        axis_line = []
        this_row = raw_img[top_line[i]: bottom_line[i]]
        this_row = cv2.resize(this_row, (int(30/this_row.shape[0]*this_row.shape[1]), 30))
        line_raw = this_row.copy()
        this_row = binary_and_resize(this_row)

        if this_row.shape[1] > 1800:
            continue
        this_row, _, length = process_img_per3pixel(this_row)
        tag = split_func(line_split_sess, this_row, length)
        tag = tag[: length]
        axis_line = post_process_per3pixel(line_raw, tag)
        axis_list.append(axis_line)

    return None, axis_list



def detect_with_pixel(img, raw_img):
    """
    输入为二值化处理后的图片地址，之后还需加入倾斜矫正。
    输出为一个二维的列表。
    列表的每个元素表示一行数据。
    每行数据由一系列方框构成，每个方框由一个列表表示，列表元素为[start_x, start_y, end_x, end_y]，四个元素代表左上和右下两个点的绝对坐标。

    和detect函数的不同之处在于这个函数同时返回了pixel list用来记录识别出来后的图片
    在原图中的像素值
    :param img_path:
    :return:
    """
    height, width = img.shape
    reversed_img = np.array([[float(255 - i) for i in j] for j in img])
    threshold = 30 * 255
    col_threshold = 2 * 255
    row_shadow = np.sum(reversed_img, axis=1)

    top_line = []
    bottom_line = []
    flag = False
    for i, sh in enumerate(row_shadow):
        if i == 0:
            continue
        if (sum(row_shadow[i: i+1]) > threshold) and (sum(row_shadow[i-1: i]) < threshold) and not flag:
            top_line.append(max(0, i-5))
            flag = True
        elif (sum(row_shadow[i: i+2]) < 2 * threshold) and (sum(row_shadow[i-2: i]) > 2 * threshold) and flag:
            bottom_line.append(i)
            flag = False
    else:
        if flag:
            bottom_line.append(height)
            flag = False

    axis_list = []

    left_margin = int(width / 200)
    right_margin = int(width / 70)

    for i in range(len(bottom_line)):
        if bottom_line[i] - top_line[i] < 10:
            continue
        axis_line = []
        flag = False
        this_row = reversed_img[top_line[i]: bottom_line[i]]
        col_shadow = np.sum(this_row, axis=0)
        for j, sh in enumerate(col_shadow[left_margin: -right_margin]):
            if (sh > col_threshold) and (col_shadow[j-1] < col_threshold) and not flag:
                axis = [j, top_line[i], bottom_line[i]]
                flag = True
            elif (sum(col_shadow[j: j+right_margin]) < col_threshold) and (sum(col_shadow[j-left_margin: j]) > col_threshold) and flag:
                if j - axis[0] > 8:
                    axis.insert(2, j)
                    axis_line.append(axis)
                flag = False
        else:
            if flag:
                axis.insert(2, width)
                axis_line.append(axis)
        axis_list.append(axis_line)

    # pic_list = [raw_img[rect[1]: rect[3], rect[0]: rect[2]] for line in axis_list for rect in line]
    pic_list = [[raw_img[rect[1]: rect[3], rect[0]: rect[2]] for rect in line] for line in axis_list]

    # recode the pixel
    pixel_list = [[rect[0] for rect in line] for line in axis_list]
    img = Image.fromarray(raw_img)
    draw = ImageDraw.Draw(img)
    for rects in axis_list:
        for rect in rects:
            draw.rectangle(rect)
    img = np.array(img)
    return img, pic_list, pixel_list


if __name__ == '__main__':
    # detect_raw_fig('data/5.jpg')
    # for file in os.listdir('data'):
    #     pic_list = detect('data/' + file)
    detect('data/test2.jpeg')
