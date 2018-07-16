#coding=utf8
import codecs
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
#import tensorlayer as tl
#由于之前没有考虑过要过多种单子的情况，这里新加一个init函数，用以指定到底该读取哪个数据
namepath = ''
def init(t='X'):
    global namepath
    if t == 'X':
        namepath = 'catenum_path.list'
    elif t=='N':
        namepath = 'Ncatenum_path.list'
def class_num():
    path_file = codecs.open(namepath,'r','utf8').readlines()
    path_file = [i.split() for i in path_file]
    path_file = set([i[1] for i in path_file])
    num_classes = len(path_file) #需要分类的类目个数
    return num_classes
def get_path_label(val=False):
    '''
    获取文件的文件名(完整路径)与对应标签的映射字典
    val:是否是读取测试目录,测试目录应当与训练目录一样,这样后续看结果才有意义
    return:{文件名1:label,文件名2:label2,...}
    '''
    pathfile = codecs.open(namepath, 'r', 'utf8').readlines()
    pathlist = []
    if val == False:
        pathlist = [i.replace('\n', '').split() for i in pathfile]
    else:
        pathlist = [i.replace('\n', '').replace('aftersplit','val/number_val').split() for i in pathfile]
    pathdict = dict((i[1], i[0]) for i in pathlist )
    print(pathdict)
    pathlabel = {}
    for label in pathdict:#对每一个label
        filelist = os.listdir(pathdict[label])
        for everyfile in filelist:#对每一个label目录下的文件
            pathlabel[pathdict[label] + everyfile] = int(label)
    return pathlabel

def get_image(image_path,height,width):  
    """
    从给定路径中读取图片，返回的是numpy.ndarray
    image_path:string, height:图像像素高度 width:图像像素宽度
    return:numpy.ndarray的图片tensor 
    """ 
    im = Image.open(image_path).convert('L')
    b = reshape(im,height,width)
    return b,image_path

def reshape(im,height,width):
    '''
    resize
    im:PIL读取图片后的Image对象
    '''
    #b = np.reshape(im, [im.size[1], im.size[0], 1])
    #b = tl.prepro.imresize(b, size=(height, width), interp='bilinear')
    b = im.resize((width,height),Image.BILINEAR)
    b = np.reshape(b,[b.size[1],b.size[0],1])
    return b

def normal(data,height,width):
    '''
    归一化
    '''
    data = data.astype(np.float32)
    mean = np.sum(data)/(height*width)
    std = np.max(data) - np.min(data)
    data = (data -mean)/std
    return data

def get(image_height,image_width,image_channel):
    '''
    获取所有的训练文件夹里的图片矩阵和其所对应的标签,这里标签和输出标签的对应在path.list文件里
    return:X[number,height,width,channel] Y[number,1]
    '''
    pathlabel = get_path_label()
    image_num = len(pathlabel)
    inx = 0
    X = np.zeros((image_num, image_height, image_width, image_channel), np.float32)
    Y = np.zeros((image_num,1),np.uint8)
    all_path = []
    for path in pathlabel:#对每一个label
        #data = sess.run(get_image(path,image_height,image_width))
        data,image_path = get_image(path,image_height,image_width)
        all_path.append(image_path)
        data = normal(data,image_height,image_width)
        label = pathlabel[path]
        X[inx,:,:,:] = data
        Y[inx,:] = label
        inx = inx+1
    print(X.shape)
    print(Y.shape)
    return X,Y,np.array(all_path)
