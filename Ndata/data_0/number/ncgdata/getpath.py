#coding=utf8
#因为以后训练数据可能有好几个...所以这里的程序负责生成类似的path.list文件,内容每行为图片文件夹根地址 图片类别数字
import codecs
out = codecs.open('numpath.list','w','utf8')
import os
namepath = '1106bak/'
namelist = os.listdir(namepath)
namelist  = sorted(namelist,key=lambda x:int(x))
rootpath = '/home/lihang/ocr/master2/data/data_0/number/'
for inx,name in enumerate(namelist):
    out.write(rootpath+namepath+name.decode('utf8')+'/'+' '+str(inx)+'\n')
    try:
        os.remove(rootpath+namepath+name.decode('utf8')+'/char')
    except Exception as e:
        pass
out.close()
