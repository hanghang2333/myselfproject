#coding=utf8
#将0也就是检查项目里的所有图片copy到一整个文件里，主要是为了做二分类使用
import os
import shutil
import re
zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
def contain_zh(word):
    '''
    判断传入字符串是否包含中文
    :param word: 待判断字符串
    :return: True:包含中文  False:不包含中文
    '''
    word = word.decode('utf8')
    global zh_pattern
    match = zh_pattern.search(word)
    return match
target_dir = '0/'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
from_dir = '1101data_add/'
path_list = os.listdir(from_dir)
path_list = [from_dir+i+'/' for i in path_list]
for i in path_list:
    if contain_zh(i):
        namelist = os.listdir(i)
        for j in namelist:
            shutil.copyfile(i+j,target_dir+j)


