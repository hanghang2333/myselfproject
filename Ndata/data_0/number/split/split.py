#coding=utf8
from splitter.splitter import split_from_url, Splitter
from splitter.char_splitter import char_split
import cv2
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorlayer as tl
import codecs
import number.getdata
import number.NumberNet


root_path = '1/'
out_path = 'out1/'
filelist = os.listdir(root_path)

num_num_classes = number.getdata.class_num()
print('numnumclass',num_num_classes)
num_image_width = 15
num_image_height = 50
num_image_channel = 1
numlabelfile = 'numnamelabel'
#初始化数字识别模型
g2 = tf.Graph()
with g2.as_default():
    sess2 = tf.Session()
    model2 = number.NumberNet.SingleNumNet(num_image_height,num_image_width,num_image_channel,0.5,num_num_classes)
    saver2 = tf.train.Saver()
    saver2.restore(sess2,'number/savedmodel/200model')

def make_name_label(labelfile):
    """
    化验单分类出来的标签是数字,这里从文件里读取对应数字应该对应的列表名称
    """
    namelabel = codecs.open(labelfile,'r','utf8').readlines()
    namelabel = [i.replace('\n','') for i in namelabel]
    if labelfile=='catenamelabel':
        namelabel = [i.split(",") for i in namelabel]
    else:
        namelabel = [i.split(' ')[0:2] for i in namelabel]#每行以空格分开后一般情况都是有两个，特殊情况是'1  '这样的会出来3个。这个指的是标签就是空格。
    namelabel_dict = {}
    for line in namelabel:
        namelabel_dict[int(line[0])] = line[1]
    return namelabel_dict
def predict_result_num(X):
    with g2.as_default():
        res = model2.predict(sess2, X)
    return res
def normal(data,height,width):
    '''
    归一化
    '''
    data = data.astype(np.float32)
    mean = np.sum(data)/(height*width)
    std = np.max(data) - np.min(data)
    data = (data -mean)/std
    return data
def predict_vector(X_in,height,width,X_count,cate):
    '''
    传入图片矩阵的
    批量后一次输入到网络里得出结果,好像模型预测批量数据和预测单个数据时间一样.
    X:numpy矩阵,矩阵的每一维度意思是:[行,列,图片高度,图片宽度](未进行归一化)
    cate:0 检查项目 1 数字 2 原始输入
    return:已经进行了reshape并且封装成[图片个数,图片高度,图片宽度,通道数]的numpy矩阵输入到模型后的结果,也是列表(单元素)和对应的原位置坐标字典
    '''
    X = np.zeros((X_count, height, width, 1), np.float32)
    inx = 0
    inx_rowcol = {}
    for row in range(len(X_in)):
        for col in range(len(X_in[row])):
            image = X_in[row][col]
            image = np.reshape(image,[image.shape[0],image.shape[1],1])#增加通道维度
            image = tl.prepro.imresize(image, size=(height, width), interp='bilinear')
            #image = image.resize((height,width),Image.BILINEAR)
            image = normal(image,height,width)
            X[inx,:,:,:] = image
            inx_rowcol[inx] = (row,col)
            inx = inx + 1
    if cate==0:#化验项
        result,acc = predict_result_cate(X)
        return result,inx_rowcol,acc
    elif cate==1:#数字
        result = predict_result_num(X)
        return result,inx_rowcol
    else:
        result = predict_result_catenum(X)
        return result,inx_rowcol
numnamelabel_dict = make_name_label(numlabelfile)
sp = Splitter('splitter/models/model', 'splitter/models/char_model3')
from tqdm import tqdm
for idx in tqdm(range(len(filelist))):
    if idx<19360:
        continue
    i = filelist[idx]
#for idx,i in tqdm(enumerate(filelist)):
    imt = Image.open(root_path+i)
    imt = np.asarray(imt)
    #res = sp.split_char(imt)
    res = char_split(imt)  
    num,_ = predict_vector([res],num_image_height,num_image_width,len(res),1)#这里是一维数组传入,传出的顺序也是按照原先的顺序,所以inx_rowcol不需要
    #numres = [numnamelabel_dict[num[k]] for k in range(len(res))]
    numres = num
    for jdx,imtt in enumerate(res):
        #print(out_path,numres[jdx])
        path = out_path+str(numres[jdx])+'/'
        if not os.path.exists(path):
            os.mkdir(path)
        cv2.imwrite(path+i+'_'+str(jdx)+'.png', imtt)
        #imtt.save(path+i+'_'+str(jdx)+'.png')
