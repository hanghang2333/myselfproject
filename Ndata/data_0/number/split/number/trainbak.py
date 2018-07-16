#coding=utf8
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import tensorflow as tf
from NumberNet import SingleNumNet
import getdata
import util
from sklearn.model_selection import KFold
import tensorlayer as tl
num_classes = getdata.class_num()#.
image_height = 50#数字图片应该普遍长宽比例是这样
image_width = 15
image_channel = 1
parser = argparse.ArgumentParser()
parser.add_argument('--Kth_fold',type=int,default=1,help='the Kth Fold')
parser.add_argument('--gpu_core',type=str,default='1',help='use which gpu core')
parser.add_argument('--logs',type=str,default='logs',help='logs path')
args = parser.parse_args()
np.random.seed(1)#固定下来随机化shuffle的序列
now = args.Kth_fold
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_core
with tf.Session() as sess:
    # 完成数据的读取，使用的是tensorflow的读取图片, 将数据集shuffle
    X, Y,all_path = getdata.get(image_height, image_width, image_channel)
    X, Y = util.shuffledata(X,Y,all_path)
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        now = now - 1
        if now == 0:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            all_path_train,all_path_test = all_path[train_index],all_path[test_index]
            print('Train: ',len(X_train))
            print('Test: ',len(X_test))    
    #建立模型训练
            model = SingleNumNet(image_height, image_width, image_channel, 0.5, num_classes)
            model.train(sess, X_train, Y_train, X_test, Y_test,201,test_path=all_path_test,logs=args.logs)
    '''
    #由于没有数据,下面以mnist为数据reshape一批数据
    from tensorflow.examples.tutorials.mnist import input_data
    def reshapeandnormal(X):
        X = np.reshape(X,[-1,28,28,1])
        X_res = []
        for i in X:
            imgint = (i*255).astype(np.int)
            imgresize = tl.prepro.imresize(imgint, size=(image_height, image_width), interp='bilinear')
            data = imgresize.astype(np.float32)
            mean = np.sum(data)/(image_height*image_width)
            std = np.max(data) - np.min(data)
            data = (data -mean)/std
            X_res.append(data)
        X_res = np.asarray(X_res)
        return X_res

    mnist = input_data.read_data_sets('./mnist', one_hot=False)  # they has been normalized to range (0,1)
    test_x = mnist.test.images
    test_y = mnist.test.labels
    train_x = mnist.train.images
    train_y = mnist.train.labels

    X_train = reshapeandnormal(train_x)
    Y_train = np.reshape(train_y,[-1,1])
    X_test = reshapeandnormal(test_x)
    Y_test = np.reshape(test_y,[-1,1])
    #终于改造完了
    '''