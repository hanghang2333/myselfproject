#coding=utf8
from __future__ import print_function
import codecs
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from getdata import get
import getdata
import util
import os
import sys
#import tensorlayer as tl
parser = argparse.ArgumentParser()
parser.add_argument('--Kth_fold',type=int,default=1,help='the Kth Fold')
parser.add_argument('--XN',type=str,default='X',help='X or N')#血常规还是尿常规，涉及到类别数目，数据读取等
args = parser.parse_args()
np.random.seed(1)#固定下来随机化shuffle的序列
now = args.Kth_fold
# 超参数
getdata.init(args.XN)
num_classes = getdata.class_num2() #需要分类的类目个数
print('classnum:',num_classes)
image_height = 20
image_width = 75
image_channel = 1
# 初始化
X, Y, all_path = get(image_height, image_width, image_channel)
# 将数据集shuffle
X, Y,all_path = util.shuffledata(X,Y,all_path)
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X):
    now = now - 1
    if now == 0:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        X_train = np.reshape(X_train,(-1,image_height*image_width))
        X_test = np.reshape(X_test,(-1,image_height*image_width))
        Y_train = np.reshape(Y_train,(-1,))
        Y_test = np.reshape(Y_test,(-1,))
        
        from sklearn import decomposition
        #pca = decomposition.PCA(n_components=100)
        pca = decomposition.IncrementalPCA(n_components=100,batch_size=128)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        #knn
        from sklearn.neighbors.nearest_centroid import NearestCentroid
        clf = NearestCentroid()
        clf.fit(X_train,Y_train)
        print(clf.score(X_test,Y_test))
        #RF
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
        clf.fit(X_train,Y_train)
        print(clf.score(X_test,Y_test))
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
        clf.fit(X_train,Y_train)
        print(clf.score(X_test,Y_test)) 
        #svm
        from sklearn import svm
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(X_train, Y_train)
        print(clf.score(X_test,Y_test))


