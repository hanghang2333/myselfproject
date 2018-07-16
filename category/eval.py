#coding=utf8
from __future__ import print_function
from sklearn.metrics import roc_auc_score
import codecs
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
#from TidyAlexnet import alexNet
import TidyAlexnet
from getdata import get
import getdata
import util
import os
import sys
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--Kth_fold',type=int,default=1,help='the Kth Fold')
parser.add_argument('--gpu_core',type=str,default='0',help='use which gpu core')
parser.add_argument('--logs',type=str,default='logs/05251/',help='logs path')
parser.add_argument('--XN',type=str,default='X',help='X or N')#血常规还是尿常规，涉及到类别数目，数据读取等
args = parser.parse_args()
np.random.seed(1)#固定下来随机化shuffle的序列
now = args.Kth_fold
# 超参数
getdata.init(args.XN)
num_classes = getdata.class_num2() #需要分类的类目个数
print('classnum:',num_classes)
image_height = 25
image_width = 150
image_channel = 1
# 初始化
#init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_core

# 初始化化验项目识别模型
g1 = tf.Graph()
with g1.as_default():
    sess1 = tf.Session()#全局load模型
    model1 = TidyAlexnet.alexNet(image_height=image_height, image_width=image_width, image_channel=1, keep_prob=0.5, classNum=num_classes)
    saver1 = tf.train.Saver()
    # 读取训练好的模型参数
    saver1.restore(sess1, args.logs+'orimodel/150orimodel')

def predict_result_cate(X):
    '''
    X:四维矩阵[图片个数,图片高度,图片宽度,通道数],已经经过了归一化
    return:预测结果列表
    '''
    with g1.as_default():
        res,acc = model1.predict(sess1, X)
    return res,acc

#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
X, Y, all_path = get(image_height, image_width, image_channel)
# 将数据集shuffle
X, Y,all_path = util.shuffledata(X,Y,all_path)
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    now = now - 1
    if now == 0:
        #X_train, X_test = X[train_index], X[test_index]
        #Y_train, Y_test = Y[train_index], Y[test_index]
        X_train,X_test,Y_train,Y_test,all_path_train,all_path_test = np.load('nparr/X_train.npy'),np.load('nparr/X_test.npy'),\
            np.load('nparr/Y_train.npy'),np.load('nparr/Y_test.npy'),np.load('nparr/all_path_train.npy'),np.load('nparr/all_path_test.npy')
        res = predict_result_cate(X_test)
        res1,p = res
        Y_test = np.reshape(Y_test,(Y_test.shape[0]))
        print(Y_test.shape,res1.shape,p.shape)
        Right = []
        ACC = 0
        for idx,i in enumerate(Y_test):
            if int(i)==int(res1[idx]):
                Right.append(1)
                ACC += 1
            else:
                Right.append(0)
        print(ACC)
        Right = np.array(Right)
        ACC = sum(Right)/Right.shape[0]
        #ACC = ACC*1.0/Y_test.shape[0]
        print('The ACC of the model: ',ACC)            
        AUC = roc_auc_score(Right,p)
        print('The AUC of the model: ',AUC)
        print('The mean score of the model:', ACC+AUC )  