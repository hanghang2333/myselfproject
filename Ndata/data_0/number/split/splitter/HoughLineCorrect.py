# -*- coding: utf-8 -*-
#coding=utf8
from __future__ import division
"""
Created on Wed Jun 14 17:23:30 2017

@author: l
"""

import numpy as np
import cv2
import subFunction


def line_correct(img):
    # 灰度处理并反转
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #gray = cv2.bitwise_not(gray)
    # 二值化
    #thresh = cv2.threshold(gray, 70 , 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(gray, 0 , 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 膨胀
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    print(element)
    dilated = cv2.dilate(thresh, element)
    dilated = cv2.dilate(dilated, element)
    dilated = cv2.dilate(dilated, element)
    # 边缘检测
    edges = cv2.Canny(dilated, 50, 150, apertureSize = 3)
    # Hough变换
    lines = cv2.HoughLines(edges, 1, np.pi/180, 118)
    # Hough直线分类，选择
    lineSum,lineCount,classfyNum = subFunction.classfiyHoughLine(lines,img, '1')
    # 所选的直线求平均，筛选，得到最终直线
    lineA,lineB,lineC,lineD = subFunction.SelectHoughLine(lineSum,lineCount,classfyNum,img,'1')
    # 求交点
    pt1 = subFunction.CalcIntersect(lineA,lineC,img)
    pt2 = subFunction.CalcIntersect(lineA,lineD,img)
    pt3 = subFunction.CalcIntersect(lineB,lineC,img)
    pt4 = subFunction.CalcIntersect(lineB,lineD,img)
    # 交点画线
    imgCopy = img.copy()
    cv2.line(imgCopy, pt1, pt2, (0,0,255), 4)
    cv2.line(imgCopy, pt3, pt4, (0,0,255), 4)
    cv2.line(imgCopy, pt1, pt3, (0,0,255), 4)
    cv2.line(imgCopy, pt2, pt4, (0,0,255), 4)
    # 透视变换
    lenx = ((pt2[0] - pt1[0]) + (pt4[0] - pt3[0])) // 2
    highy = ((pt3[1] - pt1[1]) + (pt4[1] - pt2[1])) // 2
    pts1 = np.float32([pt1,pt2,pt3,pt4])
    print(pts1)
    pts2 = np.float32([[0,0],[lenx,0],[0,highy],[lenx,highy]])
    print(pts2)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(gray, M, (lenx,highy))
    return dst

