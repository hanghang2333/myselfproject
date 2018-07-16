#coding=utf8
from __future__ import print_function
import codecs
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
#from TidyAlexnet import alexNet
import TidyAlexnet4 as TidyAlexnet
from getdatan import get
import getdatan as getdata
import util
import os
import sys
import numpy as np
import tensorflow as tf
#import tensorlayer as tl
parser = argparse.ArgumentParser()
parser.add_argument('--Kth_fold',type=int,default=1,help='the Kth Fold')
parser.add_argument('--gpu_core',type=str,default='0',help='use which gpu core')
parser.add_argument('--logs',type=str,default='logs/5_0421',help='logs path')
parser.add_argument('--XN',type=str,default='X',help='X or N')#血常规还是尿常规，涉及到类别数目，数据读取等
args = parser.parse_args()
np.random.seed(1)#固定下来随机化shuffle的序列
now = args.Kth_fold
# 超参数
getdata.init(args.XN)
num_classes = getdata.class_num2() #需要分类的类目个数
print('classnum:',num_classes)
image_height = 30
image_width = 200
image_channel = 1
# 初始化
#init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_core
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#with tf.Session() as sess:
    # 初始化所有变量
    #    sess.run(init_op)
    # 完成数据的读取，使用的是tensorflow的读取图片
    X, Y, all_path = get(image_height, image_width, image_channel)
    # 将数据集shuffle
    X, Y,all_path = util.shuffledata(X,Y,all_path)
    #indices = range(len(Y)) # indices = the number of images in the source data set
    #np.random.shuffle(indices)
    #X = X[indices]
    #Y = Y[indices]
    # 将数据区分为测试集合和训练集合
    #random_state = 33
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
    # K折交叉验证
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        now = now - 1
        if now == 0:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            all_path_train,all_path_test = all_path[train_index],all_path[test_index]
            print('Train: ',len(X_train))
            print('Test: ',len(X_test))
            model = TidyAlexnet.alexNet(image_height, image_width, image_channel, 0.5, num_classes)
            res = model.train(sess, X_train, Y_train, X_test, Y_test,split=120,num_epochs=200,
            num_count=5, test_path=all_path_test,logs=args.logs)
            print(res)
