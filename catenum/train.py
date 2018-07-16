#coding=utf8
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import tensorflow as tf
from CateNumNet import SingleNet
import getdata
import util
from sklearn.model_selection import KFold
#import tensorlayer as tl

parser = argparse.ArgumentParser()
parser.add_argument('--Kth_fold',type=int,default=1,help='the Kth Fold')
parser.add_argument('--gpu_core',type=str,default='0',help='use which gpu core')
parser.add_argument('--logs',type=str,default='logs/09271',help='logs path')
parser.add_argument('--XN',type=str,default='X',help='X or N')#血常规还是尿常规，涉及到类别数目，数据读取等
args = parser.parse_args()
np.random.seed(1)#固定下来随机化shuffle的序列
now = args.Kth_fold
getdata.init(args.XN)
num_classes = getdata.class_num()#.
image_height = 25#数字图片应该普遍长宽比例是这样
image_width = 150
image_channel = 1
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_core
with tf.Session() as sess:
    # 完成数据的读取，使用的是tensorflow的读取图片, 将数据集shuffle
    X, Y,all_path = getdata.get(image_height, image_width, image_channel)
    X, Y = util.shuffledata(X,Y,all_path)
    kf = KFold(n_splits=20)
    for train_index, test_index in kf.split(X):
        now = now - 1
        if now == 0:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            all_path_train,all_path_test = all_path[train_index],all_path[test_index]
            print('Train: ',len(X_train))
            print('Test: ',len(X_test))    
    #建立模型训练
            model = SingleNet(image_height, image_width, image_channel, 0.5, num_classes)
            model.train(sess, X_train, Y_train, X_test, Y_test,60,test_path=all_path_test,logs=args.logs)