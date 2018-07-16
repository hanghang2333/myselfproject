#coding=utf8
from __future__ import division
import os
import shutil


def main():
    root_path = 'data/chinese_data/'
    target_path = 'data/divide_data/'
    files = os.listdir(root_path)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    count = 0
    for target_file in files:
        cls = count % 20
        if not os.path.exists(target_path + str(cls)):
            os.mkdir(target_path + str(cls))
        count += 1
        shutil.copy(root_path + target_file, target_path + str(cls))
        

if __name__ == '__main__':
    main()
    
