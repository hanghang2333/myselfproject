#coding=utf8
from __future__ import print_function
from __future__ import division
import codecs
import numpy as np
import time
import TidyAlexnet
import getdata
import os
from PIL import Image
import tensorflow as tf

this_time = time.time()
# 超参数
num_classes = getdata.class_num() #需要分类的类目个数
print('num_class',num_classes)
image_height = 25
image_width = 150
image_channel = 1
# 初始化
os.environ["CUDA_VISIBLE_DEVICES"]="2"
sess = tf.Session()#全局load模型
model = TidyAlexnet.alexNet(image_height, image_width, image_channel, 0.5, num_classes)
saver = tf.train.Saver()
# 读取训练好的模型参数
saver.restore(sess, 'savedmodel/399model')
print('初始化load模型用时：%fs' % (time.time() - this_time))
print('start predicting')

def predict_result_cate(X):
    '''
    X:四维矩阵[图片个数,图片高度,图片宽度,通道数],已经经过了归一化
    return:预测结果列表
    '''
    return model.predict(sess, X)

def preprocess(X_pre,image_height,image_width):
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
        data,image_path = getdata.get_image(X_pre[i], image_height, image_width)
        data = getdata.normal(data,image_height,image_width)
        X[inx,:,:,:] = data
        inx = inx + 1
    return X

def predict_path(X_pre,image_height,image_width):
    '''
    预测结果接口
    X_pre:列表,列表的每一项需为欲预测的图片的完整路径名称
    return:列表,列表每一项为传入的对应图片的预测标签
    '''
    a = predict_result_cate(preprocess(X_pre,image_height,image_width))
    return a

def predict_vector(X_in,height,width,X_count):
    '''
    考虑到后续使用时可能不会再将中间图片保存再使用predict_path接口,这里新写一个传入图片矩阵的
    X:numpy矩阵,矩阵的每一维度意思是:[行,列,图片高度,图片宽度](未进行归一化)
    return:已经进行了reshape并且封装成[图片个数,图片高度,图片宽度,通道数]的numpy矩阵输入到模型后的结果,也是列表(单元素)
    '''
    X = np.zeros((X_count, height, width, 1), np.float32)
    inx = 0
    inx_rowcol = {}
    for row in range(len(X_in)):
        for col in range(len(X_in[row])):
            image = X_in[row][col]
            image = np.reshape(image,[image.shape[0],image.shape[1],1])#增加通道维度
            image = tl.prepro.imresize(image, size=(height, width), interp='bilinear')
            image = category.getdata.normal(image,height,width)
            X[inx,:,:,:] = image
            inx_rowcol[inx] = (row,col)
            inx = inx + 1
    result = predict_result_cate(X)
    return result,inx_rowcol

def predict(X,X_count):
    global image_height,image_width
    namelabel_dict = make_name_label(labelfile)
    result,inx_rowcol = predict_vector(X,image_height,image_width,X_count)
    res = {}
    for inx in range(len(result)):
        #res[inx_rowcol[inx]] = result[inx]
        res[inx_rowcol[inx]] = namelabel_dict[result[inx]]
    return res

def test(val_path,image_height,image_width):
    '''
    查看测试结果
    '''
    pathlabel = getdata.get_path_label(val=True)
    allcount = 0
    correctcount = 0
    for i in pathlabel:
        allcount = allcount + 1
        print('----')
        pred = predict_path([i],image_height,image_width)
        print('pred: ',pred[0],' real: ',pathlabel[i])
        if pred[0] == int(pathlabel[i]):
            correctcount = correctcount + 1
    print('acc: ',correctcount/allcount)

if __name__ =='__main__':
    #test('../../../data/val/cate_val/',image_height,image_width)
    print(predict_path(['/home/lihang/ocr/src/datapre/scripts/graytest/origin/1.jpg',
    '/home/lihang/ocr/src/datapre/scripts/graytest/origin/1.jpeg',\
    '/home/lihang/ocr/src/datapre/scripts/graytest/origin/2.jpeg',\
    '/home/lihang/ocr/src/datapre/scripts/graytest/origin/jQOCVg.jpeg'],image_height,image_width))
