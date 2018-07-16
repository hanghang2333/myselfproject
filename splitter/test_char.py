#coding=utf8
import tensorflow as tf
import os
import numpy as np
from .train_char import process_img
from .neural_splitter import NeuralSplitter
import cv2
from copy import deepcopy


def post_process(raw_img, tag):
    start_idx = []
    end_idx = []
    length = len(tag)
    flag = False
    #print(list(tag))
    for i, this_tag in enumerate(tag[:-3]):
        if not flag:
            if this_tag == 2:
                flag = True
                start_idx.append(i)
        else:
            if this_tag == 0:
                if (tag[i+1] == 1 or tag[i+2] == 1 or tag[i+3] == 1) and \
                        tag[i+1] != 2 and tag[i+2] != 2 and tag[i+3] !=2:
                    continue
                flag = False
                end_idx.append(i)
    else:
        if flag:
            end_idx.append(i)

    #print(start_idx)
    #print(end_idx)
    split_img_list = []
    for i in range(len(start_idx)):
        if end_idx[i] - start_idx[i] < 2:
            continue
        split_img_list.append(raw_img[:, max(0, start_idx[i] - 2): min(end_idx[i] + 2, length)])
    return split_img_list


if __name__ == '__main__':
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            # init and restore model.
            ns = NeuralSplitter(50, 500, is_training=False)
            saver = tf.train.Saver()
            saver.restore(sess, './models/char_model')

            # load test data
            root_path = '../../data/char_wrong'
            img_files = os.listdir(root_path)
            target_path = '../../data/test_char_split'
            if not os.path.exists(target_path):
                os.mkdir(target_path)
            import time
            start_time = time.time()
            for img_file in img_files:
                img_src = os.path.join(root_path, img_file)
                img = cv2.imread(img_src, 0)
                img = cv2.resize(img, (int(50/img.shape[0]*img.shape[1]), 50))
                if img.shape[1] > 500:
                    print(img_file)
                    continue
                img_copy = deepcopy(img)
                img, _, length = process_img(img, height=50, max_len=500)
                tag = ns.test(sess, img, length)
                tag = tag[: length]
                img = img.transpose()
                split_img_list = post_process(img_copy, tag)
                target_dir = os.path.join(target_path, img_file)
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)
                for i, split_img in enumerate(split_img_list):
                    cv2.imwrite(os.path.join(target_dir, str(i) + '.png'), split_img)
                cv2.imwrite(os.path.join(target_dir, img_file), img_copy)
            print(time.time() - start_time)


