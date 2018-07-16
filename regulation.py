#coding=utf8
import string
"""
使用一些规则去将输出结果归类.

初始化一个长度为5的数组用来存放如下内容,初始放空字符串.
需要标记的列:检查项目(第一个),检查结果,范围,单位,高低
除检查项目外还剩几个,没有了则四项全空,下面到某一步后没有了之后剩下的为空.
1.带有'-'或者'--'标记且'-'或'--'左右两边都不为空(长度至少为3或者4)且两边元素至少有一个是数字的为范围.(未搜索到则没有)
2.全部是数字的为检查结果(未搜索到则没有,搜索到后查看其以
  小数点结尾则看其后面是否也是全数字是的话则合并--这应该是切割时分开了的)
3.高低的长度必然是1,在剩下的里面搜索是否有长度为1且内容为上下箭头的(未搜索到则没有)
3.剩下的那些如果有这样几种,多余等于两串,等于一串,没有了.
没有了单位是空.
等于一个则认为是单位
多余两个则不好判断了...从后往前遍历吧,优先判断内容包含"'^/)('%"的为单位多个都有则顺序靠后的优先.没有的话则优先认为长度最长的那个为单位
"""
def str_count(s):
    '''找出字符串中的中英文、数字、标点符号个数'''

    count_en = count_dg = count_pu = 0
    s_len = len(s)
    for c in s:
        if c in string.ascii_letters:
            count_en += 1
        elif c.isdigit():
            count_dg += 1
        else:
            count_pu += 1
    return count_en,count_dg,count_pu
def containAny(seq,aset):
    '''
    查看字符串seq里是否包含有aset里的任意一个元素
    seq:字符串
    aset:想要检查的字符的set
    return: bool
    '''
    for c in aset:
        if c in seq:
            return True
    '''
    for c in seq:
         if c in aset:
                return True
    '''
    return False
def regu(strlist):
    '''
    将输入的结果规范化,即包含有[汉字,结果,范围,单位,高低]这五项目.
    strlist:输入的数组中每一项均为一个字符串,内容确定了第一个是汉字串,后面是字符串
    return: reslist:返回的也是一个list.
    '''
    n = len(strlist) #输入数组长度
    reslist = ['','','','',''] #要确定的五项 [汉字,结果,范围,单位,高低]
    tag = [1 for _ in strlist] #对输入数组的某个元素是否已经确定了属于什么
    #第一个是汉字串这个是确定的.
    reslist[0] = strlist[0]
    tag[0] = 0
    #找范围  带有'-'或者'--'标记且'-'或'--'左右两边都不为空(长度至少为3或者4)且至少有一个是数字的为范围.
    for i in range(n-1,-1,-1):#这个地方为了w1图片里的问题改为从后向前找，不知道是否会使得其他错误
        if tag[i] == 1:#该元素还没有确定
            if '--' in strlist[i]:
                inx = strlist[i].index('--')
                if inx!=0 and len(strlist[i])-2>inx and (strlist[i][inx-1].isdigit() or strlist[i][inx+2].isdigit()):#左右不为空
                    leng = len(strlist[i])
                    en_count,_,_ = str_count(strlist[i])
                    if en_count<=leng/3:#要求范围内容里包含英文字母数目不能太多，这个1/3是妥协，太小了会使得很多不算错误的无法识别出，大了可能会被字母串影响
                        tag[i] = 0
                        reslist[2] = strlist[i].replace('--','-')
                        break
            elif '-' in strlist[i]:
                #print('w',i,strlist[i])
                inx = strlist[i].index('-')
                if inx!=0 and len(strlist[i])-1>inx and (strlist[i][inx-1].isdigit() or strlist[i][inx+1].isdigit()):#左右不为空
                    leng = len(strlist[i])
                    en_count,_,_ = str_count(strlist[i])
                    #print(strlist[i],leng,en_count)
                    if en_count<=leng/3:
                        tag[i] = 0
                        reslist[2] = strlist[i]
                        break
            else:
                pass
    #找数字结果 全部是数字的为检查结果(未搜索到则没有,搜索到后查看其以小数点结尾则看其后面是否也是全数字是的话则合并--这应该是切割时分开了的)
    for i in range(n):
        numstr = ''
        if tag[i] == 1:
            tmp = strlist[i]
            tmp = tmp.replace('O','0').replace('L','1.').replace('()','0').replace('D','0').replace('W','00').replace('I','1').replace('.','').replace('T','7').replace(':','.')#如果去除O和小数点后全部都是数字则认为是数字
            if tmp[0]!='<' and tmp[0]!='>':#有些小数点图片切开后的效果不好导致识别错误#这个主要也是为了防止范围串干扰
                tmp = tmp.replace('<','').replace('>','')
            if tmp.isdigit():#是纯粹的数字字符串
                #查看是否需要接后面的.
                if i<n-1 and tag[i+1] == 1:
                    if strlist[i][-1]=='.' and strlist[i+1].isdigit():
                        numstr = strlist[i]+strlist[i+1]
                        reslist[1] = numstr
                        tag[i] = 0
                        tag[i+1] = 0
                        break
                numstr = strlist[i]
                reslist[1] = numstr
                tag[i] = 0
                break
    if reslist[1] == '':
        if len(strlist)>=2 and strlist[1]=='-':
            reslist[1] = strlist[1]
            tag[1] = 0
    
    #找高低 高低的长度必然是1,在剩下的里面搜索是否有长度为1且内容为上下箭头的(未搜索到则没有)
    for i in range(n):
        if tag[i] == 1:
            if strlist[i] == '{' or strlist[i] == '}' or strlist[i] == 'L' or strlist[i] == 'H':
                reslist[4] = strlist[i]
                tag[i] = 0
                break
    #找单位
    # 剩下的那些如果有这样几种,多余等于两串,等于一串,没有了.
    # 没有了单位是空.
    # 等于一个则认为是单位
    # 多余两个则不好判断了...1.优先判断内容包含"'^/)('%"的为单位多个都有则顺序靠后的优先.没有的话则优先认为长度最长的那个为单位
    if sum(tag) == 0:
        pass
    if sum(tag) == 1:
        for i in range(n):
            if tag[i]==1:
                reslist[3] = strlist[i]
                tag[i] = 0
                break
    if sum(tag) >=2:
        cantget = True
        danwei = set(['10^9/L','10^9/1.','%','g/L','g/1.','Pg','fL','f1.','10^12/L','10^12/1.','10^9/1','10^12/1','109/1','1012/1','109/L','109/1.','1012/L','1012/1.','g/1'])#常见单位
        for i in range(n-1,-1,-1):#先以这种方式查看是否有常见的单位，有的话直接认为这就是单位，就不用去下面那样找了
            if tag[i] == 1:
                if strlist[i] in danwei:
                    reslist[3]=strlist[i]
                    cantget = False
                    break
        if cantget:
            maxlen = 0
            maxleninx = n-1
            for i in range(n-1,-1,-1):
                if tag[i] == 1:
                    if len(strlist[i])>maxlen:
                        maxlen = len(strlist[i])
                        maxleninx = i
                    dwset = set(['f1.','fL','^','/','%','(',')','.'])
                    if containAny(strlist[i],dwset):
                        reslist[3] = strlist[i]
                        tag[i] = 0
                        break
            if reslist[3] =='':
                reslist[3] = strlist[maxleninx]
    #用一些简单的规则修正
    #[汉字,结果,范围,单位,高低]
    #单位字符串里的'1.'修改成'L'
    #数字串和范围串里面不会有'O',将其中的'O'全部替换为'0'，L替换为1. ()替换为0 D替换为0
    rest = restlist
    rest[1] = rest[1].replace('O','0').replace('L','1.').replace('()','0').replace('D','0').replace('W','00').replace('I','1').replace('T','7')
    if '<' in rest[1] and rest[1][0]!='<':#有些小数点图片切开后的效果不好导致识别错误
        rest[1] = rest[1].replace('<','.')
    if '>' in rest[1] and rest[1][0]!='>':
        rest[1] = rest[1].replace('>','.')
    if ':' in rest[1] and rest[1][0]!=':':
        rest[1] = rest[1].replace(':','.')
    rest[2] = rest[2].replace('O','0').replace('L','1.').replace('()','0').replace('D','0').replace('W','00').replace('T','7').replace('I','1').replace('^','-')
    if '-' in rest[2] and rest[2][0]!='-':
        rest[2] = rest[2].replace('<','.').replace('>','.').replace(':','.')
        #单位字符串修正
    rest[3] = rest[3].replace('1.','L').replace('f1','fL').replace('109/','10^9/').replace('1012/','10^12/').replace('{','f')
    if rest[3]=='9/L':
        rest[3]='g/L' 
    if rest[3]!='' and rest[3][-1]=='1':
        rest[3] = rest[3][0:-1]+'L'
    return rest

def reguall(twoX):
    res = []
    for i in twoX:
        res.append(regu(i))
    return res
#下面都是测试
#print(regu(['嗜碱性粒细胞绝对值','BAS0#','0.0-I','10^mL','0.00-0.10']))
#print(regu(['淋巴细胞绝对值','UmI-11#','1.03','10^9/L','1.00-4.00']))
#print regu(['白细胞','1.08','}','3.5-9.5','(10^O)'])
#print regu(['淋巴细胞数' ,'1.' ,'2' ,'1.1-:1.' ,'10^9/1.'])
#print regu(['大血小板比例','40.9', '%'])
#print regu(['中性粒细胞绝对值','4.18', '1.8--6.3' ,'109/L'])
#print regu(['白细胞总数','10.3','{','10^9/L','3.5-','9.5'])
def test(filename):
    import codecs
    content = codecs.open(filename,'r','utf8').readlines()
    content = [i.split() for i in content]
    print reguall(content)
    for i in content:
        print regu(i)[1:]
if __name__ == '__main__':
    #test('result1/1JpCSx.csv')
    pass
