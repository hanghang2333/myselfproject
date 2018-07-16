import os
import shutil
import cv2
import random
import numpy as np


def genpic():
    root_path = '../../../data/true_data'
    target_path = '../../../data/artificial_data'
    tag_path = '../../../data/tag'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    if not os.path.exists(tag_path):
        os.mkdir(tag_path)
    dirs = os.listdir(root_path)
    for pic_dir in dirs:
        pic_dir_path = os.path.join(root_path, pic_dir)
        pic_names = os.listdir(pic_dir_path)


        # generate 20 pic per line
        for i in range(10):
            # read the first picture
            concat_pic = cv2.imread(os.path.join(pic_dir_path, pic_names[0]), 0)
            # get the dim of the picture
            pic_dim = concat_pic.shape[0]

            concat_pic = cv2.resize(concat_pic, (int(30/pic_dim*concat_pic.shape[1]), 30))
            # the tag of the picture
            tag = [2 if j == 0 else 1 for j in range(concat_pic.shape[1])]
            # add the other pic piece
            for pic_name in pic_names[1:]:
                pic_src = os.path.join(pic_dir_path, pic_name)
                pic = cv2.imread(pic_src, 0)
                pic = cv2.resize(pic, (int(30/pic_dim*pic.shape[1]), 30))
                print(pic.shape)
                rand = random.randint(0, 30)
                concat_pic = np.concatenate((concat_pic, 255 - np.zeros((30, rand), dtype=np.uint8), pic), axis=1)
                tag += [0 for _ in range(rand)]
                tag += [2 if j == 0 else 1 for j in range(pic.shape[1])]
            tag = np.array(tag)
            np.save(os.path.join(tag_path, pic_dir + '_' + str(i) + '.npy'), tag)
            cv2.imwrite(os.path.join(target_path, pic_dir + '_' + str(i) + '.png'), concat_pic)


def concat_tag():
    root_path = '../../../data/tag'
    target_path = '../../../data/concat_tag'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    tags = os.listdir(root_path)
    for tag_file in tags:
        tag_path = os.path.join(root_path, tag_file)
        tag = np.load(tag_path)
        pad = len(tag) % 3
        tag = np.concatenate((tag, np.zeros([3 - pad])), axis=0)
        new_tag = []
        for idx in range(int(len(tag) / 3)):
            target_tag = tag[idx * 3: idx * 3 + 3]
            if 2 in target_tag:
                new_tag.append(2)
            elif 1 in target_tag:
                new_tag.append(1)
            else:
                new_tag.append(0)
        new_tag = np.array(new_tag)
        print(new_tag)
        output_path = os.path.join(target_path, tag_file)
        np.save(output_path, new_tag)


if __name__ == '__main__':
    # genpic()
    concat_tag()
