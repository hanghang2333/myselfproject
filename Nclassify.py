#coding=utf8
from __future__ import print_function
from __future__ import division
import codecs
import numpy as np
import time
import category.TidyAlexnet
import number.NumberNet
import catenum.CateNumNet
import category.getdata
from PIL import Image
import tensorflow as tf
#import tensorlayer as tl
from splitter import splitter
from splitter import Npic_classifier as pic_classifier
from splitter.splitter import split_from_url, Splitter
import cv2
import os
from get_crf_result import reguall

def make_name_label(labelfile):
    """
    化验单分类出来的标签是数字,这里从文件里读取对应数字应该对应的列表名称
    """
    namelabel = codecs.open(labelfile,'r','utf8').readlines()
    namelabel = [i.replace('\n','') for i in namelabel]
    if labelfile=='Ncatenamelabel':
        namelabel = [i.split(",") for i in namelabel]
    else:
        namelabel = [i.split(' ')[0:2] for i in namelabel]
    namelabel_dict = {}
    for line in namelabel:
        namelabel_dict[int(line[0])] = line[1]
    return namelabel_dict

catenamelabel_dict = make_name_label('Ncatenamelabel')
catenumnamelabel_dict = make_name_label('Ncatenumnamelabel')
numnamelabel_dict = make_name_label('Nnumnamelabel')
cate_image_height,cate_image_width,num_image_height,num_image_width=25,150,50,15

sp = Splitter('splitter/models/model', 'splitter/models/char_model3')

this_time = time.time()
# 化验项目部分的超参数
cate_num_classes = len(catenamelabel_dict) #需要分类的化验项目类目个数
print('catenumclass',cate_num_classes)
# 初始化化验项目识别模型
g1 = tf.Graph()
with g1.as_default():
    sess1 = tf.Session()#全局load模型
    model1 = category.TidyAlexnet.alexNet(image_height=cate_image_height, image_width=cate_image_width, image_channel=1, keep_prob=0.5, classNum=cate_num_classes)
    saver1 = tf.train.Saver()
    # 读取训练好的模型参数
    saver1.restore(sess1, 'category/Nsavedmodel/1101')

#数字英文部分的超参数
num_num_classes = len(numnamelabel_dict)
print('numnumclass',num_num_classes)
#初始化数字识别模型
g2 = tf.Graph()
with g2.as_default():
    sess2 = tf.Session()
    model2 = number.NumberNet.SingleNumNet(image_height=num_image_height, image_width=num_image_width, image_channel=1, keepPro=0.5, classNum=num_num_classes)
    saver2 = tf.train.Saver()
    saver2.restore(sess2,'number/Nsavedmodel/1106')

#初始化前置二分类器模型,使用的超参数大部分和化验项部分一致
catenum_num_classes = len(catenumnamelabel_dict) #需要分类的化验项目类目个数
print('catenumnumclass',catenum_num_classes)
g3 = tf.Graph()
with g3.as_default():
    sess3 = tf.Session()
    model3 = catenum.CateNumNet.SingleNet(image_height=cate_image_height, image_width=cate_image_width, image_channel=1, keepPro=0.5, classNum=catenum_num_classes)
    saver3 = tf.train.Saver()
    saver3.restore(sess3,'catenum/Nsavedmodel/1101')
print('初始化load模型用时：%fs' % (time.time() - this_time))
print('start predicting')


def predict_result_cate(X):
    '''
    X:四维矩阵[图片个数,图片高度,图片宽度,通道数],已经经过了归一化
    return:预测结果列表
    '''
    with g1.as_default():
        res,acc = model1.predict(sess1, X)
    return res,acc[0]

def predict_result_num(X):
    with g2.as_default():
        res = model2.predict(sess2, X)
    return res

def predict_result_catenum(X):
    with g3.as_default():
        res = model3.predict(sess3,X)
    return res

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
            #image = np.reshape(image,[image.shape[0],image.shape[1],1])#增加通道维度
            #image = tl.prepro.imresize(image, size=(height, width), interp='bilinear')
            image = Image.fromarray(image.astype(np.uint8))
            image = image.resize((width,height),Image.BILINEAR)
            image = np.reshape(image,[image.size[1],image.size[0],1])
            image = category.getdata.normal(image,height,width)
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

def predict_before(X):
    if isinstance(X,bool):#切割split_from_url()没有切割成功会raise异常,返回值测试了下是True,也就是二值类型
        return None
    else:
        #先判断检查结果还是检查项目
        X_count = 0#切割出来的图片的数目
        for i in X:
            for j in i:
                X_count = X_count + 1
        #print(X_count)
        global cate_image_height,cate_image_width,catenumlabelfile
        result,inx_rowcol = predict_vector(X,cate_image_height,cate_image_width,X_count,2)
        import copy
        X_bak = copy.deepcopy(X)
        for inx in range(len(result)):
            X_bak[inx_rowcol[inx][0]][inx_rowcol[inx][1]] = int(catenumnamelabel_dict[result[inx]])
        return X_bak

def printfig2(X,url):
    #未写好
    import matplotlib.pyplot as plt
    fig2 = plt.figure(2)
    height = len(X)
    maxwidth = 0 
    for x in X:
        width = 0
        for idx,i in enumerate(x):
            if idx == 0:
                width += 1
            else:
                width += len(i)
        maxwidth = max(maxwidth,width)
    maxwidth = maxwidth+20
    print(maxwidth)     
    #width = max([len(x) for x in X])
    for inxrow,row in enumerate(X):
        nowcol = 0
        for inxcol,col in enumerate(row):
            if not isinstance(col,list):
                now = col
                plt.subplot2grid((height,maxwidth),(inxrow,nowcol),colspan=4)
                im = Image.fromarray(now.astype(np.uint8))
                plt.imshow(im,cmap="gray")
                plt.axis("off")
                nowcol += 4
            else:
                now = col
                for i in now:
                    plt.subplot2grid((height,maxwidth),(inxrow,nowcol))
                    im = Image.fromarray(i.astype(np.uint8))
                    plt.imshow(im,cmap="gray")
                    plt.axis("off")
                    nowcol += 1          
                nowcol +=2
    fig2.show()
    fig2.savefig(url+'-2.jpg')
    plt.close(2)
def printfig1(X,url):
    import matplotlib.pyplot as plt
    fig1 = plt.figure(1)
    height = len(X)
    maxwidth = max([len(x) for x in X])
    for inxrow,row in enumerate(X):
        nowcol = 0
        for inxcol,col in enumerate(row):
            now = col
            plt.subplot2grid((height,maxwidth),(inxrow,inxcol))
            im = Image.fromarray(now.astype(np.uint8))
            plt.imshow(im,cmap="gray")
            plt.axis("off")
            nowcol += 1
    fig1.show()
    fig1.savefig(url+'-1.jpg')
    plt.close(1)
def predict_after(X,url):
    #首先进行二次分割,after_two_split里需要调用predict_brefore(X)来获取二分类的结果
    #after_two_split为邵文良spliiter部分需要提供的一个函数
    X_bak = predict_before(X)
    if X_bak == None:
        print(u'可能是切割异常')
        return None
    XX = pic_classifier.classify_pic(X,X_bak,sp.split_char)
    #printfig1(X,url)
    #printfig2(XX,url)
    global cate_image_height,cate_image_width,num_image_height,num_image_width
    #将X_bak传回得到新的数组XX
    #XX每一行第一个是化验项矩阵,后面每个数组依次是字符串内容
    res = []
    #inxrow = 0
    for row in XX:
        tmp = []
        for i,c in enumerate(row):
            if isinstance(c,list):
                num,_ = predict_vector([row[i]],num_image_height,num_image_width,len(row[i]),1)#这里是一维数组传入,传出的顺序也是按照原先的顺序,所以inx_rowcol不需要
                numres = [numnamelabel_dict[num[k]] for k in range(len(row[i]))]
                numres = ''.join(numres)
                tmp.append(numres)
            else:
                cate,_,acc = predict_vector([[row[i]]],cate_image_height,cate_image_width,1,0)
                tmp.append(catenamelabel_dict[cate[0]])
        res.append(tmp)
    return res

def predict_url(url):
    res =  predict_after(sp.split_img(url),url)
    #为了观察方便将结果写入表格(非结构化表格,可能还需要进一步完善定义规则之类)
    try:
        if res == None:
            print(url)
            return res
        filename = os.path.splitext(os.path.basename(url))
        out = codecs.open('Nresult/'+filename[0]+'.csv','w','utf8')
        for row in res:
            out.write(' '.join(row)+'\n')
        out.close()
    except Exception as e:#写入文件出错就不管了,这不是主要
        pass
    return res

def test(root,src):
    out = codecs.open(root+src+'.res','w','utf8')#将归类的结果写入到文件里
    src = root+src
    res = predict_url(src)
    if res!=None:
        #res=reguall(res)
        for row in res:
            out.write(','.join(row)+'\n')
            for i in row:
                print(i+' ', end='')
            print('')
    else:
        print('无法切割')
import regulation
if __name__ =='__main__':
    root = 'Njieguo/'
    imlist = os.listdir(root)
    for i in imlist:
        src = i
        if i[-4:]=='jpeg':
            test(root,src)
    
