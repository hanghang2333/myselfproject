#coding=UTF8
import os,codecs
import sys
print(sys.path)
label2label_path = 'label2label.list'
label2label = codecs.open(label2label_path,'r','utf8').readlines()
label2label = [i.replace('\n','') for i in label2label]
label2label = [i.split(',') for i in label2label]
label2label = [[i[0],i[1].replace(' ','')] for i in label2label]#给的大类标签有的大类标签内容里有空格
label2 = sorted(list(set([i[1] for i in label2label])))#排序仅仅是想要每次生成的catenamelabel一样(不保证)
label1 = [i[0] for i in label2label]
count = [0 for i in range(len(label2))]
for idx,i in enumerate(label2label):
    num = 0
    try:
        num = len(os.listdir(os.path.join(r"E:\ocr\master\data\data_0\cate_data\1025",i[0])))
    except Exception:
        pass
    nowidx = label2.index(i[1])
    count[nowidx]+=num
label_count_dict = dict()
for idx,i in enumerate(label2):
    label_count_dict[i] = count[idx]
label_count_dict = sorted(label_count_dict.items(),key=lambda x:x[1],reverse=True)
print(label_count_dict[0:10],sum(count))

k = 20
topk = [label_count_dict[i][0] for i in range(k)]
out = codecs.open('label2label_plot.list','w','utf8')
out1 = codecs.open('catenamelabel_plot.list','w','utf8')
for line in label2label:
    if line[1] in topk:
        out.write(line[0]+','+line[1]+'\n')
for i in range(k):
    out1.write(str(i)+','+topk[i]+'\n')