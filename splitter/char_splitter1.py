#coding=utf8
from __future__ import division
import numpy as np
import cv2
import os
from preprocess import resize, binary
from PIL import Image

def char_split(pic):
    resize_height = 50
    pic = resize(pic, target_height=resize_height)
    pic_copy = pic.copy()
    pic = binary(pic)
    # 列切割
    reversed_img = np.array([[float(255 - i) for i in j] for j in pic])
    threshold = 1 * 255
    flag = False
    shadow = np.sum(reversed_img, axis=0)
    start_idx = []
    end_idx = []
    for i, ax in enumerate(shadow):
        if not flag:
            if ax >= threshold:
                flag = True
                start_idx.append(i)
        else:
            if ax < threshold:
                flag = False
                end_idx.append(i)
    else:
        if flag:
            end_idx.append(i)

    # if start_idx == end_idx, we determine it a bad pic,and we would remove it
    splitted_pic = [pic_copy[:, start_idx[i]: end_idx[i]] for i in range(len(start_idx)) if start_idx[i] != end_idx[i]]
    return splitted_pic


def main():
    root_path = 'data/number_data/'
    dirs = os.listdir(root_path)
    # for current_dir in dirs:
    #     files = os.listdir(os.path.join(root_path, current_dir))
    #     for file in files:
    #         pic = cv2.imdecode(np.fromfile(root_path + current_dir + '/' + file, dtype=np.uint8), 0)
    #         char_split(pic)

    for target_dir in dirs:
        pic = cv2.imdecode(np.fromfile(root_path + target_dir, dtype=np.uint8), 0)
        char_split(pic)

def test():
    path = '1.jpg'
    im = Image.open(path).convert('L')
    b = np.reshape(im, [im.size[1], im.size[0]])
    res = char_split(b)
    for i in range(len(res)):
        imt = Image.fromarray(res[i])
        imt.save(path+str(i)+'.jpeg')


if __name__ == '__main__':
    test()
