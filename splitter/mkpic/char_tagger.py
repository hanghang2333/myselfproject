import tkinter as tk
import os
import shutil
import time
import cv2


def remove_block():
    root_path = '../../../data/biaozhu/imageblock1'
    target_path = '../../../data/biaozhu/imageblock3'
    if not os.path.exists((target_path)):
        os.mkdir(target_path)
    img_file = os.listdir(root_path)
    for file in img_file:
        img = cv2.imread(os.path.join(root_path, file))
        new_file = ''.join(file.split('.')[:2]) + '.jpg'
        print(new_file)
        cv2.imwrite(os.path.join(target_path, new_file), img)



if __name__ == '__main__':
    remove_block()
    root_path = '../../../data/biaozhu/imageblock3'
    true_path = '../../../data/char_true'
    wrong_path = '../../../data/char_wrong'
    if not os.path.exists(true_path):
        os.mkdir(true_path)
    if not os.path.exists(wrong_path):
        os.mkdir(wrong_path)
    dirs = os.listdir(root_path)

    def true_split(event):
        global top
        global target_dir
        global target_path
        global pre_dir

        print(target_path)
        if event.char == '1':
            shutil.move(target_path, true_path)
            pre_dir = target_dir
            top.quit()
        elif event.char == '2':
            shutil.move(target_path, wrong_path)
            top.quit()
        elif event.char == '3':
            os.remove(true_path + '/' + pre_dir)

    import pandas as pd
    tags = pd.read_csv('../../../data/biaozhu/biaozhu.csv', names=['1'])['1']
    tag_dic = {''.join(i.split('<+++>')[0].split('.')[: 2]) + '.png': i.split('<+++>')[1] for i in tags}
    import pickle
    with open('tag.dict', 'wb') as dict_out:
        pickle.dump(tag_dic, dict_out)
    print(tag_dic)

    for target_dir in dirs:
        top = tk.Toplevel()
        top['background'] = 'black'
        target_path = os.path.join(root_path, target_dir)
        pic = tk.PhotoImage(file=target_path)
        tk.Label(top, image=pic).pack()

        # 讲解
        tk.Label(top, text=tag_dic[target_dir]).pack()
        text = tk.Label(top, text='按1正确分割，按2错误分割')
        text.pack()
        # 设定键盘输入
        top.bind('<Key>', true_split)
        top.mainloop()
