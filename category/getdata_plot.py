import codecs
import os
import numpy as np
#from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
#import tensorlayer as tl

dataroot = "1025/"
catename = 'catenamelabel_plot.list'
label2label_path = 'label2label_plot.list'


def get_path_label_2(rootpath):
    #进行二次归类后数据的读取方式彻底修改
    #首先读取类与类对应表得到原始目录名称和对应大类名称。然后对大类排序后对应到1-n这样的数字标签上。
    #而后对读取的图片直接标注为对应的数字标签
    label2label = codecs.open(label2label_path,'r','utf8').readlines()
    label2label = [i.replace('\n','') for i in label2label]
    label2label = [i.split(',') for i in label2label]
    label2label = [[i[0],i[1].replace(' ','')] for i in label2label]#给的大类标签有的大类标签内容里有空格
    label2 = sorted(list(set([i[1] for i in label2label])))#排序仅仅是想要每次生成的catenamelabel一样(不保证)
    label1 = [i[0] for i in label2label]
    l2l_dict = {}
    l2lidx_dict = {}
    l2lidx_dict_re = {}
    for idx,i in enumerate(label2):
        l2lidx_dict[i]=idx
        l2lidx_dict_re[idx] = i
    labelnamefile = codecs.open(catename,'w','utf8')#现在每个数字对应具体什么标签内容无法从原始数据得到，这样的话在读取时生成这个文件
    for i in range(len(l2lidx_dict)):
        labelnamefile.write(str(i)+','+l2lidx_dict_re[i]+'\n')
    labelnamefile.close()
    for i in label2label:
        l2l_dict[i[0]] = i[1]
    pathlabel = {}
    for label in label1:#对每一个label
        '''
        try:
            #filelist = os.listdir(rootpath.encode('utf8')+label.encode('utf8'))
            filelist = os.listdir(os.path.join(rootpath.encode('utf8'),label.encode('utf8')))
            #print(filelist[0])
            for everyfile in filelist:#对每一个label目录下的文件
                pathlabel[os.path.join(rootpath.encode('utf8'),label.encode('utf8'),everyfile)] = l2lidx_dict[l2l_dict[label]]
        except Exception:
            print(rootpath.encode('utf8')+label.encode('utf8'))#有些在大类文件里的小类实际上数据里并没有对应的文件夹
        '''
        try:
            filelist = os.listdir(rootpath.encode('utf8')+label.encode('utf8'))
            for everyfile in filelist:#对每一个label目录下的文件
                pathlabel[rootpath.encode('utf8')+label.encode('utf8')+b'/'+ everyfile] = l2lidx_dict[l2l_dict[label]]
        except Exception:
            print(rootpath.encode('utf8')+label.encode('utf8'))#有些在大类文件里的小类实际上数据里并没有对应的文件夹
    
    #因为二次归类的读取数据是在程序里进行的，所以有时候可能想看下原始多少类以及大类多少类这样的
    return pathlabel
def class_num2():
    if not os.path.exists(catename):#不存在说明还没有生成。这个地方确实很不方便，不过除非数据类别保证不变了，目前也只能如此
        get_path_label_2(dataroot)
    path_file = codecs.open(catename,'r','utf8').readlines()
    path_file = [i.split(',') for i in path_file]
    path_file = set([i[1] for i in path_file])
    num_classes = len(path_file) #需要分类的类目个数
    return num_classes
#get_path_label_2('/home/lihang/ocr/data/tagedimages/0_add/')
#print(class_num2())


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
    #b = np.reshape(im, [im.size[1], im.size[0], 1])#image对象里先是宽度后是高度
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
    if std==0:
        std = 1
    data = (data -mean)/std
    return data

def get(image_height,image_width,image_channel):
    '''
    获取所有的训练文件夹里的图片矩阵和其所对应的标签,这里标签和输出标签的对应在path.list文件里
    return:X[number,height,width,channel] Y[number,1]
    '''
    #pathlabel = get_path_label()
    pathlabel = get_path_label_2(dataroot)
    image_num = len(pathlabel)
    inx = 0
    X = np.zeros((image_num, image_height, image_width, image_channel), np.float32)
    Y = np.zeros((image_num),np.uint16)
    all_path = []
    for path in pathlabel:#对每一个label
        #data = sess.run(get_image(path,image_height,image_width))
        data,image_path = get_image(path,image_height,image_width)
        all_path.append(image_path)
        data = normal(data,image_height,image_width)
        label = pathlabel[path]
        X[inx,:,:,:] = data
        Y[inx] = label
        inx = inx+1
    print(X.shape)
    print(Y.shape)
    return X,Y,np.array(all_path)

#X, Y ,path= get(30,100,1)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1, random_state=33)
