import os
import cv2
import sys
sys.path.append('../')
import pickle
import numpy as np


def tag_data():
    """
    用来对原数据进行标注
    :return:
    """

    from char_splitter import char_split_for_train
    root_path = '../../../data/char_true'
    img_files = os.listdir(root_path)

    # load tag dict
    with open('tag.dict', 'rb') as tag_input:
        tag_dict = pickle.load(tag_input)

    pic_target = '../../../data/block_img'
    if not os.path.exists(pic_target):
        os.mkdir(pic_target)
    tag_target = '../../../data/block_tag'
    if not os.path.exists(tag_target):
        os.mkdir(tag_target)

    for img_file in img_files:

        img = cv2.imread(os.path.join(root_path, img_file), 0)
        img, start_idx, end_idx = char_split_for_train(img)
        tag = tag_dict[img_file]
        if len(tag) != len(start_idx):
            continue
        tag_list = [0 for _ in range(img.shape[1])]
        for i, j in zip(start_idx, end_idx):
            tag_list[i] = 2
            for idx in range(i+1, j):
                tag_list[idx] = 1
        cv2.imwrite(os.path.join(pic_target, img_file), img)
        np.save(os.path.join(tag_target, img_file.split('.')[0] + '.npy'), np.array(tag_list))


def transform_tag():
    root_path = '../../../data/block_tag'
    target_path = '../../../data/block_tag_three'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    tag_files = os.listdir(root_path)
    for tag_file in tag_files:
        tag = np.load(os.path.join(root_path, tag_file))
        start_idx= []
        for i, idx in enumerate(tag[: -2]):
            if idx == 2:
                start_idx.append(i)
        for i in start_idx:
            tag[i: i+3] = 2
        print(tag)
        np.save(os.path.join(target_path, tag_file), tag)


def add_bias():
    root_path = '../../../data/block_img'
    img_files = os.listdir(root_path)
    tag_path = '../../../data/block_tag'
    target_path = '../../../data/block_img_bias'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    for img_file in img_files[:]:
        img_path = os.path.join(root_path, img_file)
        print(np.load(os.path.join(tag_path, img_file.split('.')[0] + '.npy')))

        img = cv2.imread(img_path, 0)
        highy, lenx = img.shape
        for i in range(0, int(lenx / 13), int(lenx / 30) + 1):
            transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [lenx-i, 0], [i, highy], [lenx, highy]]),
                                                    np.float32([[0, 0], [lenx, 0], [0, highy], [lenx, highy]]))
            bias_img = cv2.warpPerspective(img, transform, (lenx, highy))
            cv2.imwrite(os.path.join(target_path, img_file.split('.')[0] + '_' + str(i) + '.png'), bias_img)
            transform = cv2.getPerspectiveTransform(np.float32([[i, 0], [lenx, 0], [0, highy], [lenx-i, highy]]),
                                                    np.float32([[0, 0], [lenx, 0], [0, highy], [lenx, highy]]))
            bias_img = cv2.warpPerspective(img, transform, (lenx, highy))
            cv2.imwrite(os.path.join(target_path, img_file.split('.')[0] + '_' + str(-i) + '.png'), bias_img)


def add_blur():
    """
    add blur in img
    """
    root_path = '../../../data/block_img_bias'
    target_path = '../../../data/block_img_bias_blur'
    img_files = os.listdir(root_path)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    for img_file in img_files[:]:
        img_path = os.path.join(root_path, img_file)
        img = cv2.imread(img_path, 0)
        # find background color
        img_list = np.concatenate(img)
        img_list = sorted(img_list)
        background_color = img_list[-int(len(img_list) / 3)]
        black_color = img_list[10]
        for i in range(0, background_color - black_color, 25):
            new_img = []
            for x in img:
                new_x = []
                for y in x:
                    new_x.append(min(background_color-20, y+i) if y < background_color - 20 else y)
                new_img.append(new_x)
            new_img = np.array(new_img)
            cv2.imwrite(os.path.join(target_path, img_file.split('.')[0] + '_' + str(i) + '.png'), new_img)


def count_len():
    root_path = '../../../data/block_img_bias_blur'
    img_files = os.listdir(root_path)
    max_len = 0
    for img_file in img_files:
        x, y = cv2.imread(os.path.join(root_path, img_file), 0).shape
        if y > max_len:
            max_len = y
    print(max_len)


if __name__ == '__main__':
    # tag_data()
    # add_bias()
    # add_blur()
    # count_len()
    transform_tag()
