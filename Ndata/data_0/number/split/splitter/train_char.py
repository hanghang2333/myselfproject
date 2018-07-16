#coding=utf8
import tensorflow as tf
import os
import numpy as np
from .neural_splitter import NeuralSplitter
from PIL import Image


def process_img(img, height=30, max_len=1200, tag=None):
    img = img / 255
    length = img.shape[1]
    img = np.concatenate((img, np.ones([height, max_len - length])), axis=1)
    img = np.transpose(img)
    img = 1 - img
    if isinstance(tag, np.ndarray):
        tag = np.concatenate((tag, np.ones([max_len - length, ])), axis=0)
    return img, tag, length


def load_data(img_path, tag_path):
    """
    Read the image data and tag.
    Then padding the image and tag.
    """
    img_list = []
    tag_list = []
    length_list = []
    img_files = os.listdir(img_path)
    for img_file in img_files:
        img_src = os.path.join(img_path, img_file)
        tag_src = os.path.join(tag_path, '_'.join(img_file.split('.')[0].split('_')[:-2]) + '.npy')
        img = Image.open(img_src)
        img = np.array(img)
        tag = np.load(tag_src)
        img, tag, length = process_img(img, height=50, max_len=500, tag=tag)
        img_list.append(img)
        tag_list.append(tag)
        length_list.append(length)
    img_list = np.array(img_list)
    tag_list = np.array(tag_list)
    length_list = np.array(length_list)
    return img_list, tag_list, length_list


if __name__ == '__main__':
    img_list, tag_list, length_list = load_data('../../data/block_img_bias_blur', '../../data/block_tag_three')
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            ns = NeuralSplitter(50, 500)
            tf.global_variables_initializer().run()
            ns.train(sess, img_list, tag_list, length_list, num_epochs=100)


