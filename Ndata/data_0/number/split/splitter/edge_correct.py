#coding=utf8
from __future__ import division
import cv2
import numpy as np
import os
from .preprocess import preprocess
from PIL import Image
import time
import math


def hough_line_detect(img, raw_img):
    """
    输入图片，进行直线识别
    :param img:图片矩阵
    :return:
    """

    height, width = img.shape
    img_copy = img.copy()
    result = np.zeros([height, width])
    # 边缘检测
    img = cv2.Canny(img, 50, 150)
    # 上下膨胀，迭代三次
    img = cv2.dilate(img, np.uint8([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), iterations=7)
    # 概率直线检测
    lines = cv2.HoughLinesP(img, 1, np.pi/180, int(width * 0.3), minLineLength=width * 0.4,  maxLineGap=2)
    if isinstance(lines, np.ndarray):
        lines = np.reshape(lines, [-1, 4])

        # 对统一同一直线进行连接处理，同时过滤竖线
        # 每个line包含x1, y1, x2, y2, k, b
        new_lines = []

        def is_same(line1, k, b):
            # 判断是否是同一条直线
            if abs(line1[0] * k + b - line1[1]) / math.sqrt(k * k + 1) < 15 or \
                                    abs(line1[2] * k + b - line1[3]) / math.sqrt(k * k + 1) < 10:
                return True
            else:
                return False

        for line in lines:
            # 过滤竖线
            if abs(line[2] - line[0]) > abs(line[3] - line[1]):
                if line[0] > line[2]:
                    line[0], line[1], line[2], line[3] = line[2], line[3], line[0], line[1]
                # 计算直线方程 y = kx + b 中的k和b
                for i, new_line in enumerate(new_lines):
                    if is_same(line, new_line[4], new_line[5]):
                        if new_line[0] > line[0]:
                            new_lines[i][0] = line[0]
                            new_lines[i][1] = line[1]
                        if new_line[2] < line[2]:
                            new_lines[i][2] = line[2]
                            new_lines[i][3] = line[3]
                        new_lines[i][4] = (new_lines[i][3] - new_lines[i][1]) / (new_lines[i][2] - new_lines[i][0])
                        new_lines[i][5] = new_lines[i][3] - new_lines[i][4] * new_lines[i][2]
                        break
                else:
                    k = (line[3] - line[1]) / (line[2] - line[0])
                    b = line[3] - k * line[2]
                    new_lines.append([line[0], line[1], line[2], line[3], k, b])

        # 如果只有一条直线，
        # TODO 可能需要改为异常处理
        if len(new_lines) < 2:
            return [img], [raw_img], False
        # 取出间距最大的两条直线
        new_lines.sort(key=lambda x: x[1])
        start_line = 0
        max_len = 0
        for i, line in enumerate(new_lines[:-1]):
            if new_lines[i+1][1] - line[1] > max_len:
                max_len = new_lines[i+1][1] - line[1]
                start_line = i

        # 找到两条目标线
        line1 = new_lines[start_line]
        line2 = new_lines[start_line + 1]
        # cv2.line(result, (line1[0], line1[1]), (line1[2], line1[3]), (255), 1)
        # cv2.line(result, (line2[0], line2[1]), (line2[2], line2[3]), (255), 1)

        # 目标线延长至矩形
        pt1x = line1[0]
        pt1y = line1[1]
        pt2x = line1[2]
        pt2y = line1[3]
        pt3x = line2[0]
        pt3y = line2[1]
        pt4x = line2[2]
        pt4y = line2[3]

        len_l1 = ((pt2y - pt1y) ** 2 + (pt2x - pt1x) ** 2) ** 0.5
        len_l2 = ((pt4y - pt3y) ** 2 + (pt4x - pt3x) ** 2) ** 0.5
        if len_l1 > len_l2 * 1.2:
            k1 = (pt2y - pt1y) / (pt2x - pt1x) + 0.001
            rk1 = - 1 / k1
            b2 = pt1y - rk1 * pt1x
            b3 = pt2y - rk1 * pt2x

            k2 = (pt3y - pt4y) / (pt3x - pt4x) + 0.001
            b1 = pt3y - k2 * pt3x
            pt3x = int((b2 - b1) / (k2 - rk1))
            pt3y = int(k2 * pt3x + b1)
            pt4x = int((b3 - b1) / (k2 - rk1))
            pt4y = int(k2 * pt4x + b1)

        elif len_l2 > len_l1 * 1.2:
            pt1x, pt2x, pt1y, pt2y, pt3x, pt4x, pt3y, pt4y = pt3x, pt4x, pt3y, pt4y, pt1x, pt2x, pt1y, pt2y

            k1 = (pt2y - pt1y) / (pt2x - pt1x) + 0.001
            rk1 = - 1 / k1
            b2 = pt1y - rk1 * pt1x
            b3 = pt2y - rk1 * pt2x

            k2 = (pt3y - pt4y) / (pt3x - pt4x) + 0.001
            b1 = pt3y - k2 * pt3x

            pt3x = int((b2 - b1) / (k2 - rk1))
            pt3y = int(k2 * pt3x + b1)
            pt4x = int((b3 - b1) / (k2 - rk1))
            pt4y = int(k2 * pt4x + b1)
            pt1x, pt2x, pt1y, pt2y, pt3x, pt4x, pt3y, pt4y = pt3x, pt4x, pt3y, pt4y, pt1x, pt2x, pt1y, pt2y

        lenx = (abs(pt2x - pt1x) + abs(pt4x - pt3x)) // 2
        highy = (abs(pt3y - pt1y) + abs(pt4y - pt2y)) // 2
        transform = cv2.getPerspectiveTransform(np.float32([[pt1x, pt1y], [pt2x, pt2y], [pt3x, pt3y], [pt4x, pt4y]]),
                                    np.float32([[0, 0], [lenx, 0], [0, highy], [lenx, highy]]))
        img_copy = cv2.warpPerspective(img_copy, transform, (width, height))
        # 截取
        img_copy = img_copy[:highy, :lenx]
        raw_img = cv2.warpPerspective(raw_img, transform, (width, height))
        # 截取
        raw_img = raw_img[:highy, :lenx]
    else:
        return [img], [raw_img], False

    # 竖线检测与消除
    new_img = img_copy.copy()
    img_copy = cv2.Canny(img_copy, 50, 150)
    # 左右膨胀，迭代三次
    img_copy = cv2.dilate(img_copy, np.uint8([[0, 0, 0], [1, 1, 1], [0, 0, 0]]), iterations=3)

    # img_copy = cv2.dilate(img_copy, np.uint8([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), iterations=2)

    # 概率直线检测
    lines = cv2.HoughLinesP(img_copy, 1, np.pi/180, int(highy * 0.1), minLineLength=highy * 0.4, maxLineGap=3)
    if isinstance(lines, np.ndarray):
        lines = np.reshape(lines, [-1, 4])
        for line in lines:
            if abs(line[0] - line[2]) < lenx / 10 and abs(line[0] - lenx/2) < lenx / 6:
                new_img = [new_img[:, :min(line[2], line[0])], new_img[:, max(line[2], line[0]):]]
                raw_img = [raw_img[:, :min(line[2], line[0])], raw_img[:, max(line[2], line[0]):]]
                return new_img, raw_img, True
    return [new_img], [raw_img], True


def yc_correct_by_syn(img):
    """
    基于游程编码的算法，由沈燕妮的算法重写为python版本
    :param img: 输入的image矩阵
    :return:
    """
    # 获取高度和宽度
    height, width = img.shape
    start_time = time.time()
    # 生成黑色的图片
    img_copy = 255. - np.zeros([height, width])
    line_recd = np.zeros([height, 2])

    # 逐行扫描
    for i, line in enumerate(img):
        cnt = 0
        j = 0
        while j < width:
            if img[i][j] == 0:
                cnt += 1
            else:
                # 搜索下一行左边是否是黑色
                if cnt > 0 and i+1 < height and img[i+1][j-cnt] == 0:
                    k = j - cnt - 1
                    while k >= 0:
                        if img[i+1][k] != 0:
                            break
                        cnt += 1
                        k -= 1
                # 同理向右搜索
                if cnt > 0 and i+1 < height and img[i+1][j] == 0:
                    while j < width:
                        if img[i+1][j] != 0:
                            break
                        cnt += 1
                        j += 1

                # 将本行中最长的游程保存
                if cnt > 0 and line_recd[i][1] > 0 and j-cnt-line_recd[i][1] <= 2:
                    line_recd[i][1] = j-1
                elif cnt > line_recd[i][1] - line_recd[i][0] + 1:
                    line_recd[i][0] = j - cnt
                    line_recd[i][1] = j - 1
                cnt = 0
            j += 1
        # 黑白作图观察
        # if line_recd[i][0] == 0 and line_recd[i][1] == 0:
        #     continue
        # else:
        #     cv2.line(img_copy, (int(line_recd[i][0]), i), (int(line_recd[i][1]), i), 255)

    # 游程连接
    lines = np.zeros([height, 4], dtype=np.int32)
    line_num = 0
    for i in range(height):
        if line_recd[i][0] == 0 and line_recd[i][1] == 0:
            continue
        # 连接条件：比如游程AB和游程CD，min(B,D)-max(A,C)>=-1，且所属行相差<=5
        if line_num > 0 and min(line_recd[i][1], lines[line_num][2]) - max(line_recd[i][0], lines[line_num][0]) >= -1 \
            and abs(i - max(lines[line_num][1], lines[line_num][3])) <= 5:
            if line_recd[i][0] < lines[line_num][0] and abs(i - lines[line_num][1]) <= 5:
                lines[line_num][0] = line_recd[i][0]
                lines[line_num][1] = i
            if line_recd[i][1] > lines[line_num][2] and abs(i - lines[line_num][3]) <= 5:
                lines[line_num][2] = line_recd[i][1]
                lines[line_num][3] = i
        else:
            line_num += 1
            lines[line_num][0] = line_recd[i][0]
            lines[line_num][1] = i
            lines[line_num][2] = line_recd[i][1]
            lines[line_num][3] = i

    # 全部长度
    lines_len = np.zeros([line_num])
    for i in range(line_num):
        lines_len[i] = np.sqrt((lines[i][0] - lines[i][2]) ** 2 + (lines[i][1] - lines[i][3]) ** 2)

    # 长度排序
    line_pair = [(lines_len[i], i) for i in range(line_num)]
    line_pair.sort(key=lambda x: x[0], reverse=True)

    # 取最长的五条作为目标行
    target_num = 5
    target_line = []
    for i in range(target_num):
        target_line.append(lines[line_pair[i][1]])

    # 选取直线

    for i in range(target_num):
        for j in range(i + 1, target_num):
            if line_pair[j][0] > line_pair[0][0] * 0.75 and \
                            abs(target_line[i][1] - target_line[j][1]) > height * 0.35:
                if target_line[i][1] > target_line[j][1]:
                    line1 = target_line[i]
                    line2 = target_line[j]
                else:
                    line1 = target_line[j]
                    line2 = target_line[i]
                break

    pt1x = line2[0]
    pt1y = line2[1]
    pt2x = line2[2]
    pt2y = line2[3]
    pt3x = line1[0]
    pt3y = line1[1]
    pt4x = line1[2]
    pt4y = line1[3]

    lenx = (abs(pt2x - pt1x) + abs(pt4x - pt3x)) // 2
    highy = (abs(pt3y - pt1y) + abs(pt4y - pt2y)) // 2
    # img_copy = cv2.line(img_copy, (line1[0], line1[1]), (line1[2], line1[3]), 0)
    # img_copy = cv2.line(img_copy, (line2[0], line2[1]), (line2[2], line2[3]), 0)
    transform = cv2.getPerspectiveTransform(np.float32([[pt1x, pt1y], [pt2x, pt2y], [pt3x, pt3y], [pt4x, pt4y]]),
                                np.float32([[0, 0], [lenx, 0], [0, highy], [lenx, highy]]))
    img = cv2.warpPerspective(img, transform, (height, width))
    #
    img = img[:highy, :lenx]
    return img


if __name__ == '__main__':

    # for file in os.listdir('data'):
    #     img = cv2.imread('data/' + file)
    #     img = preprocess(img)
    #     img = np.array([[float(255 - i) for i in j] for j in img])
    #     height, width = img.shape
    #     img = cv2.resize(img, (int(800 / height * width), 800))
    #     hough_line_detect(img)
    img = cv2.imread('data/' + 'test2.jpeg')
    img = preprocess(img)
    height, width = img.shape
    # img = cv2.resize(img, (int(1600 / height * width), 1600))
    # yc_correct_by_syn(img)
    hough_line_detect(img)
