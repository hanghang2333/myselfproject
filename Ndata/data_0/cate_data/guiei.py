#coding=utf8
#二次归类。将标注后的检查项目类(约一千以上)归类到大类里。使用字符串匹配方式。
#生成一个文件，内容两列，分别为原始标签-归类标签。这张表其实由专业人员来完成比较好。目前先使用程序生成。
#字符串匹配不可可能完善，生成的文件需要手动调整。也就是对于某一类归错了，手动修改一下。
#思路
'''
一千多个类按字符串归大类。首先要知道有哪些大类，而这个无法从程序统计里确切知道,只能大概知道。
标签里有英文并且有的标签只有英文，还有的确实只有英文部分不同但确实是两个意思，去除英文的话会错误归为一类。
导致直接去除英文去比较汉字也不合适

1.所有标签去除空格和数字。如果带有括号并且括号内不包含中文则去除整个括号，否则只去除括号本身。
2.取内容全部是汉字的作为是大类A(这个肯定不完善，不过也只有这种方法。。)
3.对所有标签，如果A中有一个标签被完全包含在其中并且长度相差不能太多，将这个A中的标签作为该标签上层标签。
4.如果第3步没有找到，则对当前标签K，对A中每一个大类标签里的标签I长度为n，查看是否I中有n-1个字符完全在k中出现并且长度不能相差太多，有的话则以I为标签。
5.上一步不用编辑距离主要是因为k的长度可能会比真实大类长很多，编辑距离的话阈值无法设定。
6.如果还是找不到大类则将其自身标签作为标签(也就相当于自己就是大类)
7,将每一个标签和其对应的大类写到文件里去
8.手动观察是否有错误并手动修改
'''
import os
import codecs
import re
zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
def contain_zh(word):
    '''
    判断传入字符串是否包含中文
    :param word: 待判断字符串
    :return: True:包含中文  False:不包含中文
    '''
    word = word
    global zh_pattern
    match = zh_pattern.search(word)
    return match

def cleanstr(s):
    s = s.replace(u'（','(').replace(u'）',')')
    #处理不规则标签的初步方法。去除数字，一部分括号内部，所有括号
    r = set(list('0123456789.'))
    s = filter(lambda x:x not in r,s)#删除数字
    if '(' in s and ')' in s:
        start = s.index('(')
        end = s.index(')')
        flag = True
        if end>start:
            for i in range(start,end+1):
                if ord(s[i])>128:
                    flag = False
                    break
        if flag:
            s = s[0:start]+s[end+1:]
    s = s.replace('(','').replace(')','').replace('*','').replace(u'◇','').replace(u'★','').replace(' ','').replace('-','')
    return s
            

def getdalei(nowlabel):
    nowlabel = [cleanstr(s) for s in nowlabel]#将所有标签都处理下
    nowlabel = set(nowlabel)#去重
    ret = []
    for i in nowlabel:
        flag = True
        for j in i:
            if not u'\u4e00'<=j<=u'\u9fff':
                flag = False
        if flag:
            ret.append(i)
    #for i in ret:
    #    print i
    #print(len(ret))
    ret.sort(key=lambda x:len(x),reverse=True)
    return ret
def makedict(nowlabel,daleilabel):
    retd = {}
    for i in nowlabel:
        tmp = cleanstr(i)
        done = False
        for j in daleilabel:
            if j in tmp and len(j)>=int((1.0*len(tmp)/3*2)):
                retd[i] = j
                done = True
                break
        if not done:
            for j in daleilabel:
                n = len(j)
                count = 0
                for cr in j:
                    if cr in tmp:
                        count+=1
                if count>=n-1 and len(j)>=int((1.0*len(tmp)/3*2)):
                    retd[i]=j
                    done = True
        if not done:
            #print(tmp)
            retd[i]=tmp
    return retd
def test():
    alllabel = os.listdir('1101data/')
    alllabel = [i.decode('utf8') for i in alllabel]
    alllabel = filter(lambda x:contain_zh(x),alllabel)
    dalei = getdalei(alllabel)
    retd = makedict(alllabel,dalei)
    daleinow = set(retd.values())
    #for i in daleinow:
    #    print(i)
    print(len(daleinow))
    outfile = codecs.open('label2label.txt','w','utf8')
    for i in retd:
        outfile.write(i+'<+-+>'+retd[i]+'\n')
    outfile.close()
test()
