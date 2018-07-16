#coding=utf8
from __future__ import division
from skimage import io
import cv2
from .preprocess import binary, resize
from .edge_correct import hough_line_detect
from .border_detector import detect, detect_by_hough, detect_by_network
from .neural_splitter import NeuralSplitter
import tensorflow as tf
from .train import process_img
from .test_char import post_process
from copy import deepcopy


def split_from_url(img_src):
    """
    将一副图片中表格里的所有项识别出来。
    :param img_path: 图片的路径
    :return: 二维的list,每个元素是一张图片。
    """

    # 转为灰度图
    img = io.imread(img_src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_copy = img.copy()

    # 二值化
    img = binary(img, 10)

    # 统一尺寸
    img = resize(img, target_height=1200)
    img_copy = resize(img_copy, target_height=1200)

    # 边缘检测
    # img = yc_correct_by_syn(img)
    imgs, img_copy, flag = hough_line_detect(img, img_copy)

    if not flag:
        print(u'由于某些原因，图片无法切割')
        # raise Exception('由于某些原因，图片无法切割')
        return False
    # for i in range(len(img_copy)):
    #     Image.fromarray(imgs[i]).show()
    # 去噪

    # 投影法切割
    # 判断是双列还是单列
    if len(img_copy) == 1:
        imgs = imgs[0]
        img_copy = img_copy[0]
        img, pic_list = detect_by_hough(imgs, img_copy)
    else:
        img, pic_list = detect_by_hough(imgs[0], img_copy[0])
        img2, pic_list2 = detect_by_hough(imgs[1], img_copy[1])
        pic_list += pic_list2

    return pic_list

class Splitter(object):
    """
    this is the class that used to split img by nerual network
    """
    def __init__(self, line_split_model_path, char_split_model_path):
        # init the line split model
        self.line_split_graph = tf.Graph()
        with self.line_split_graph.as_default():
            self.line_split_sess = tf.Session()
            self.line_split_model = NeuralSplitter(90, 600, is_training=False)
            self.line_saver = tf.train.Saver()
            self.line_saver.restore(self.line_split_sess, line_split_model_path)

        self.char_split_graph = tf.Graph()
        with self.char_split_graph.as_default():
            self.char_split_sess = tf.Session()
            self.char_split_model = NeuralSplitter(50, 500, is_training=False)
            self.char_saver = tf.train.Saver()
            self.char_saver.restore(self.char_split_sess, char_split_model_path)


    def split_img(self, img_path):
        # 转为灰度图
        img = io.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_copy = img.copy()

        # 二值化
        img = binary(img, 10)

        # 统一尺寸
        img = resize(img, target_height=1200)
        img_copy = resize(img_copy, target_height=1200)

        # 边缘检测
        # img = yc_correct_by_syn(img)
        imgs, img_copy, flag = hough_line_detect(img, img_copy)

        if not flag:
            print(u'由于某些原因，图片无法切割')
            # raise Exception('由于某些原因，图片无法切割')
            return False
        # for i in range(len(img_copy)):
        #     Image.fromarray(imgs[i]).show()
        # 去噪

        # 投影法切割
        # 判断是双列还是单列
        with self.line_split_graph.as_default():
            if len(img_copy) == 1:
                imgs = imgs[0]
                img_copy = img_copy[0]
                _, pic_list = detect_by_network(imgs, img_copy, self.line_split_model.test, self.line_split_sess)
            else:
                _, pic_list = detect_by_network(imgs[0], img_copy[0], self.line_split_model.test, self.line_split_sess)
                _, pic_list2 = detect_by_network(imgs[1], img_copy[1], self.line_split_model.test, self.line_split_sess)
                pic_list += pic_list2

        return pic_list

    def split_char(self, img):
        with self.char_split_graph.as_default():
            img = cv2.resize(img, (int(50*1.0/img.shape[0]*img.shape[1]), 50))
            if img.shape[1] > 500:
                return []
            img_copy = deepcopy(img)
            img, _, length = process_img(img, height=50, max_len=500)
            tag = self.char_split_model.test(self.char_split_sess, img, length)
            tag = tag[: length]
            split_img_list = post_process(img_copy, tag)
        return split_img_list

