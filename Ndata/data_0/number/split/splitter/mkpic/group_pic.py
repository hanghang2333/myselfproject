import os
import shutil


def group_check_item():
    item_path1 = '../../../data/tagedimages/0'
    item_path2 = '../../../data/tagedimages/0_add'
    target_path = '../../../data/split_data'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    item_path_list = [item_path1]
    for item_path in item_path_list:
        for item_dir in os.listdir(item_path):
            if os.path.isdir(os.path.join(item_path, item_dir)):
                for file in os.listdir(os.path.join(item_path, item_dir)):
                    print(file)
                    target_dir = '_'.join(file.split('_')[1: 3])
                    path = os.path.join(target_path, target_dir)
                    if not os.path.exists(path):
                        os.mkdir(path)
                    shutil.copy(os.path.join(item_path, item_dir, file), path)
            else:
                shutil.copy(os.path.join(item_path, item_dir), target_path)


def exclude_other():
    other_path = '../../../data/tagedimages/2'
    target_path = '../../../data/split_data'
    other_dir = ['_'.join(file.split('_')[1: 3]) for file in os.listdir(other_path)]
    other_dir = list(set(other_dir))
    total_dir = os.listdir(target_path)
    for i in other_dir:
        if i in total_dir:
            shutil.rmtree(os.path.join(target_path, i))


def add_nonitem():
    root_path = '../../../data/tagedimages/1'
    target_path = '../../../data/split_data'
    root_file = os.listdir(root_path)
    target_dir = os.listdir(target_path)
    for file in root_file:
        if '_'.join(file.split('_')[1: 3]) in target_dir:
            shutil.move(os.path.join(root_path, file), os.path.join(target_path, '_'.join(file.split('_')[1: 3])))


def remove_less():
    target_path = '../../../data/split_data'
    target_dirs = os.listdir(target_path)
    for target_dir in target_dirs:
        path = os.path.join(target_path, target_dir)
        if len(os.listdir(path)) < 3:
            shutil.rmtree(path)


def batch_binary():
    root_path = '../../../data/split_data'
    target_path = '../../../data/binary_split_data'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    import cv2
    dirs = os.listdir(root_path)
    for target_dir in dirs:
        line_path = os.path.join(root_path, target_dir)
        line_target = os.path.join(target_path, target_dir)
        if not os.path.exists(line_target):
            os.mkdir(line_target)
        pic_srcs = os.listdir(line_path)
        for pic_src in pic_srcs:
            pic = cv2.imread(os.path.join(line_path, pic_src), 0)
            pic = cv2.adaptiveThreshold(pic, 255,
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                        thresholdType=cv2.THRESH_BINARY,
                                        blockSize=21,
                                        C=10)
            cv2.imwrite(os.path.join(line_target, pic_src.split('.')[0] + '.png'), pic)


if __name__ == '__main__':
    # group_check_item()
    # exclude_other()
    # add_nonitem()
    # remove_less()
    batch_binary()