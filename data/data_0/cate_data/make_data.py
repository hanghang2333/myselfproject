#coding=utf8
import argparse
import os
import shutil
from PIL import Image
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str,default='1025/',help='the path of image folder')
parser.add_argument('--mn',type=str,default='n',help='Multi Or Num')
parser.add_argument('--m',type=int,default='1',help='Num*M')
parser.add_argument('--n',type=int,default=10,help='Num')
args = parser.parse_args()
path = args.path
mn = args.mn
m = args.m
n = args.n

from keras.preprocessing.image import ImageDataGenerator
def get_generator(featurewise_center=False, featurewise_std=False,
                  rotation=3, width_shift=0.05, height_shift=0.05,
                  zoom=[0.9, 1.1], horizontal=False, vertical=False):
    '''
    图片数据随机化生成器定义，具体含义参看keras文档
    '''
    datagen = ImageDataGenerator(
        featurewise_center=featurewise_center,
        featurewise_std_normalization=featurewise_std,
        rotation_range=rotation,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        zoom_range=zoom,
        horizontal_flip=horizontal,
        vertical_flip=vertical)
    return datagen
datagen = get_generator()
def make_one_dir(tpath):
    imagelist = os.listdir(tpath)
    num_file = len(imagelist)
    num_gen = 0
    if mn == 'm':#如果是生成几倍于原数目的图片
        num_gen = m*num_file
    if mn == 'n':
        num_gen = n
    inx = 0
    for i in range(num_gen):
        if inx == num_file:
            inx = 0
        now = imagelist[inx]
        now = tpath+now
        img = Image.open(now).convert('L')
        img = np.reshape(img, [img.size[1], img.size[0], 1])
        for x,y in datagen.flow(np.array([img]),np.array([[1]]),batch_size=1,save_to_dir=tpath):
            break
        inx = inx + 1

import re
zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
def contain_zh(word):
    '''
    判断传入字符串是否包含中文
    :param word: 待判断字符串
    :return: True:包含中文  False:不包含中文
    '''
    word = word.decode('utf8')
    global zh_pattern
    match = zh_pattern.search(word)
    return match

def make_all_dir(root_path,target):
    dir_path = os.listdir(root_path)
    for now_path in dir_path:
        tpath = root_path+now_path+'/'
        numfile = len(os.listdir(tpath))
        if numfile<target:
            make_one_dir(tpath)
make_all_dir(path,n)
