import tkinter as tk
import os
import shutil
import time


if __name__ == '__main__':
    root_path = '../../../data/binary_split_data'
    true_path = '../../../data/true_data'
    wrong_path = '../../../data/wrong_data'
    if not os.path.exists(true_path):
        os.mkdir(true_path)
    if not os.path.exists(wrong_path):
        os.mkdir(wrong_path)
    dirs = os.listdir(root_path)

    def true_split(event):
        global top
        global target_path
        print(target_path)
        if event.char == '1':
            shutil.move(target_path, true_path)
            top.quit()
        elif event.char == '2':
            shutil.move(target_path, wrong_path)
            top.quit()

    for target_dir in dirs:
        top = tk.Toplevel()
        top['background'] = 'black'
        target_path = os.path.join(root_path, target_dir)
        pic_srcs = os.listdir(target_path)
        img_list = []
        path_list = []
        for pic_src in pic_srcs:
            pic_path = os.path.join(target_path, pic_src)
            path_list.append(pic_path)
            img_list.append(tk.PhotoImage(file=pic_path))

        for i in range(len(img_list)):
            tk.Label(top, image=img_list[i]).pack()

        # 讲解
        text = tk.Label(top, text='按1正确分割，按2错误分割')
        text.pack()
        # 设定键盘输入
        top.bind('<Key>', true_split)
        top.mainloop()
