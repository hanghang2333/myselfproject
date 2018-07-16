#coding=utf8
from __future__ import division
import cv2
import time
import numpy as np


def preprocess(img):
    # 黑白化
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, 6)

    # 二值化
    bw_img = cv2.adaptiveThreshold(img, 255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY,
                                   blockSize=21,
                                   C=20)
    return bw_img


def border_detect(img):
    """
    边缘检测
    :param img: 输入的图片
    :return: 边缘检测后的图片
    """
    img = cv2.Canny(img, 50, 150)
    return img


def binary(img, C=10):
    """
    将图片进行二值化处理，二值化的阈值为周围的所有像素的均值加C
    :param img: 输入的图片矩阵
    :param C: 二值化参数，阈值为均值加C
    :return: 二值化后的图片
    """
    # 判断是否是二值化后的图片

    bw_img = cv2.adaptiveThreshold(img, 255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY,
                                   blockSize=21,
                                   C=15)
    return bw_img


def resize(img, target_height=800):
    height, width = img.shape
    img = cv2.resize(img, (int(target_height / height * width), target_height))
    return img

if __name__ == '__main__':
    # 读取图像
    img = cv2.imread('data/6blVtb.jpeg')
    img = np.array(img)

    # 黑白化
    bw_img = cv2.cvtColor(img, 6)

    # 二值化
    bw_img = cv2.adaptiveThreshold(bw_img, 255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY,
                                   blockSize=21,
                                   C=30)

    # 旋转图像
    # coords = np.column_stack(np.where(bw_img < 1))
    # print(coords)
    # angle = -cv2.minAreaRect(coords)[-1]
    # print(angle)
    # h, w = bw_img.shape[:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # routed = cv2.warpAffine(bw_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 输出图像
    # cv2.imwrite('data/6.png', bw_img)

    # 显示图像
    # cv2.imshow('2', bw_img)
    # cv2.waitKey(0)
    from PIL import Image
    bw_img = Image.fromarray(bw_img)
    bw_img.show()
