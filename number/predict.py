#coding=utf8
from __future__ import print_function
import codecs
import numpy as np
import time
import NumberNet
import getdata
import os
import tensorflow as tf
from PIL import Image

this_time = time.time()
# 超参数
num_classes = 11
image_height = 30
image_width = 30
image_channel = 1
# 初始化
os.environ["CUDA_VISIBLE_DEVICES"]="2"
sess = tf.Session()#全局load模型
model = NumberNet.SingleNumNet(image_height, image_width, image_channel, 0.5, num_classes)
saver = tf.train.Saver()
# 读取训练好的模型参数
saver.restore(sess, 'savedmodel/10model')
print('初始化用时：%fs' % (time.time() - this_time))
print('start predicting')

def predict_result(X):
    '''
    X:四维矩阵[图片个数,图片高度,图片宽度,通道数],已经经过了归一化
    return:预测结果列表
    '''
    return model.predict(sess, X)

def preprocess(X_pre,image_height=25,image_width=100):
    '''
    输入图片也需要进行与训练时相同的预处理才能放到模型里传播
    这里假设X_pre是图片的完整路径列表list,可以不只一张图片
    X_pre:列表,列表的每一项需为欲预测的图片的完整路径名称
    return:四维矩阵[图片个数,图片高度,图片宽度,通道数],已经经过了归一化
    '''
    numbers = len(X_pre)
    X = np.zeros((numbers, image_height, image_width, image_channel), np.float32)
    inx = 0
    for i in range(numbers):
        data = getdata.get_image(X_pre[i], image_height, image_width)
        data = getdata.normal(data,image_height,image_width)
        X[inx,:,:,:] = data
        inx = inx + 1
    return X

def predict(X_pre):
    '''
    预测结果接口
    X_pre:列表,列表的每一项需为欲预测的图片的完整路径名称
    return:列表,列表每一项为传入的对应图片的预测标签
    '''
    a = predict_result(preprocess(X_pre))
    return a

def predict_vector(X,height=30,width=30):
    '''
    考虑到后续使用时可能不会再将中间图片保存再使用predict接口,这里新写一个传入图片矩阵的
    X:numpy矩阵,矩阵的每一维度意思是:[图片高度,图片宽度](未进行归一化)
    return:已经进行了reshape并且封装成[图片个数,图片高度,图片宽度,通道数]的numpy矩阵输入到模型后的结果,也是列表(单元素)
    '''
    im = Image.fromarray(np.uint8(X))
    b = getdata.reshape(im,height,width)
    X = np.zeros((1, height, width, 1), np.float32)
    X[0,:,:,:] = b
    a  = predict_result(X)
    return a


def test(val_path):
    '''
    查看测试结果
    '''
    pathlabel = getdata.get_path_label(val=True)
    allcount = 0
    correctcount = 0
    for i in pathlabel:
        allcount = allcount + 1
        print('----')
        pred = predict([i])
        print('pred: ',pred[0],' real: ',pathlabel[i])
        if pred[0] == int(pathlabel[i]):
            correctcount = correctcount + 1
    print('acc: ',correctcount/allcount)

if __name__ =='__main__':
    test(val_path = '../../../data/val/number_val/')

