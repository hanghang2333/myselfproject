# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:52:51 2017

@author: l
"""
import os
import cv2
import pymysql
import pandas as pd

# 连接sql
conn = pymysql.connect(host='rdsiffbiuir7f3u.mysql.rds.aliyuncs.com', port=3306, user='ai_user', passwd='Th0B8ZiA', db='ai', charset='utf8')
cursor = conn.cursor() 

# 从sql获取已挑选的血常规图片的url和caseimageid
sqlcmd="SELECT a.* FROM ai.tcaseimage a, ai.taxonomy_index b where a.caseimageid = b.content_id order by a.caseimageid desc"
rowNums = cursor.execute(sqlcmd) # 血常规图片个数 
print('查询的行数为：' + str(rowNums)) 
a=pd.read_sql(sqlcmd,conn)
imgs_url=a['imageurl'] # 全部血常规图片的url
imgs_caseimageid=a['caseimageid'] # 全部血常规图片的id

# 全部血常规单子逐个保存、切割、将切割的碎片录入sql
for i in range(rowNums):
    # 读取图片，存入文件夹pic中
    img_src=imgs_url[i]    
    img_name = img_src.split('/') # 血常规图片名称
    cap = cv2.VideoCapture(img_src)
    if( cap.isOpened() ) :
        ret,img = cap.read()
        img_path='pic'
        isExists=os.path.exists(img_path)
        if not isExists:
            os.mkdir(img_path)
        cv2.imwrite(img_path + '/' + img_name[-1], img) # 将血常规图片存入/pic
        
    # 切割，全部切割后图片同存入文件夹seg中
    segnum = 3 # 切割所得碎片个数，此处为假设，待合并【切割代码】
    # ……
    
    # 将本次切割的图片录入表image_regional
    n=4 # 切割图片的id，从当前列表末尾接着递增
    caseid = imgs_caseimageid[i] # 原血常规图片的id    
    for k in range(segnum): 
        filename = '' # 本切割碎片的文件名，应从【切割代码】中获取
    #files = os.listdir('seg')
    #for filename in files: 
        insertcmd = "INSERT INTO image_regional(id,caseimageid,url) VALUES (" + str(n) + "," + str(caseid) + ",'" + filename + "')"
        cursor.execute(insertcmd)
        conn.commit()
        n=n+1
    
# 关闭sql
cursor.close()
conn.close()