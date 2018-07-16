#coding=utf8
import tensorflow as tf
import os
import numpy as np
from .train import process_img, process_img_per3pixel
from .neural_splitter import NeuralSplitter
import cv2


def binary_and_resize(img):
    img = cv2.adaptiveThreshold(img, 255,
                                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                thresholdType=cv2.THRESH_BINARY,
                                blockSize=21,
                                C=10)
    return img


def post_process_per3pixel(raw_img, tag):
    start_idx = []
    end_idx = []
    flag = False
    for i, this_tag in enumerate(tag):
        if not flag:
            if this_tag != 0:
                flag = True
                start_idx.append(i)
        else:
            if this_tag != 1 and tag[i-1] != 1:
                flag = False
                end_idx.append(i)
    else:
        if flag:
            end_idx.append(i)

    split_img_list = []
    for i in range(len(start_idx)):
        if end_idx[i] - start_idx[i] < 3:
            continue
        split_img_list.append(raw_img[:, start_idx[i] * 3: end_idx[i] * 3])
    return split_img_list


def post_process(raw_img, tag):
    start_idx = []
    end_idx = []
    flag = False
    for i, this_tag in enumerate(tag):
        if not flag:
            if this_tag != 0:
                flag = True
                start_idx.append(i)
        else:
            if this_tag != 1 and tag[i-1] != 1:
                flag = False
                end_idx.append(i)
    else:
        if flag:
            end_idx.append(i)

    split_img_list = []
    for i in range(len(start_idx)):
        if end_idx[i] - start_idx[i] < 8:
            continue
        split_img_list.append(raw_img[:, start_idx[i]: end_idx[i]])
    return split_img_list


def get_line_splitter():
    line_split_sess = tf.Session()
    with tf.device('/cpu:0'):
        # init and restore model.
        ns = NeuralSplitter(90, 600, is_training=False)
        saver = tf.train.Saver()
        saver.restore(line_split_sess, './models/model')
        def get_split_result(img):
            img = cv2.resize(img, (int(30/img.shape[0]*img.shape[1]), 30))
            raw_img = img.copy()
            img = binary_and_resize(img)
            if img.shape[1] > 1800:
                return None
            img, _, length = process_img_per3pixel(img)
            tag = ns.test(line_split_sess, img, length)
            tag = tag[: length]
            split_img_list = post_process_per3pixel(raw_img, tag)
            return split_img_list
        return get_split_result


def abandoned():
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            # init and restore model.
            ns = NeuralSplitter(90, 600, is_training=False)
            saver = tf.train.Saver()
            saver.restore(sess, './models/model')

            # load test data
            root_path = '../../data/line_data'
            img_files = os.listdir(root_path)
            target_path = '../../data/test_split_results'
            if not os.path.exists(target_path):
                os.mkdir(target_path)

            for img_file in img_files:
                img_src = os.path.join(root_path, img_file)
                img = cv2.imread(img_src, 0)
                img = cv2.resize(img, (int(30/img.shape[0]*img.shape[1]), 30))
                raw_img = img.copy()
                img = binary_and_resize(img)
                if img.shape[1] > 1800:
                    print(img_file)
                    continue
                img, _, length = process_img_per3pixel(img)
                tag = ns.test(sess, img, length)
                tag = tag[: length]
                split_img_list = post_process_per3pixel(raw_img, tag)
                target_dir = os.path.join(target_path, img_file)
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)
                for i, split_img in enumerate(split_img_list):
                    cv2.imwrite(os.path.join(target_dir, str(i) + '.png'), split_img)
                cv2.imwrite(os.path.join(target_dir, img_file), raw_img)


if __name__ == '__main__':
    root_path = '../../data/line_data'
    img_files = os.listdir(root_path)
    target_path = '../../data/test_split_results'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    splitter = get_line_splitter()
    for img_file in img_files:
        img_src = os.path.join(root_path, img_file)
        img = cv2.imread(img_src, 0)
        split_img_list = splitter(img)
        print(split_img_list)

