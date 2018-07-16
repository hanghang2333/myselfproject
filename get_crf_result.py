#coding=utf8

import CRFPP
import sys,re,codecs


def extract_entities(label, X, input):
    entities = [input[0],'','','','']
    re_entity = re.compile(r'(A+)|(B+)|(C+)|(D+)')
    m = re_entity.search(label)
    while m:
        entity_labels = m.group()
        start_index_char = label.find(entity_labels)
        entity = ''.join(i for i in X[start_index_char:start_index_char + len(entity_labels)])
        entities['ABCD'.find(entity_labels[0])+1] = entity
        label = list(label)
        label[start_index_char:start_index_char + len(entity_labels)] = ''.join('O' for i in entity_labels)
        label = ''.join(label)
        m = re_entity.search(label)
    return entities


def get_recognized_entity(input,  is_result_empty,model_path = './model/model_all'):
    try:
        # -v 3: access deep information like alpha,beta,prob
        # -nN: enable nbest output. N should be >= 2
        tagger = CRFPP.Tagger("-m "+ model_path + " -v 3 -n2")

        # clear internal context
        tagger.clear()
        
        # add context
        str_origin = ''
        chunk_index_list = []
        for index, item in enumerate(input[1:]):
            str_origin += item
            for _ in item:
                chunk_index_list.append(index)
        assert len(str_origin) == len(chunk_index_list)
        for i,j in zip(str_origin, chunk_index_list):        
            tagger.add(i + ' ' + str(j))

        # parse and change internal stated as 'parsed'
        tagger.parse()
        
        size = tagger.size()
        xsize = tagger.xsize()
        label_list = []
        for i in range(0, size):
           for j in range(0, (xsize - 1)):
            label_list.append(tagger.y2(i))

        label_str = ''.join(i for i in label_list)
        
        entities = extract_entities(label_str, str_origin, input)
        if entities[1] == '' and not is_result_empty:
            input_list = []
            input_list.append(input[0])
            for i in input[2:]:
                input_list.append(i)
            #print input_list
            return get_recognized_entity(input_list,is_result_empty = True)
        else:
            return entities

    except RuntimeError, e:
        print "RuntimeError: ", e,

def reguall(twoX):
    res = []
    for i in twoX:
        #print(i)
        i = [k.encode('utf8') for k in i]
        #print(i)
        rest = get_recognized_entity(i,is_result_empty = False)
        #下面是依据先验进行修正
        #1.将结果和范围字符串里的TIO修正为710,()为0，D为0
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

        rest = [k.decode('utf8') for k in rest]
        res.append(rest)
    return res

if __name__ == '__main__':
    '''  
    cnt = 1
    with codecs.open('./X/yIR03A.csv','r') as f1:
        for i in f1.readlines():
            #print cnt,i.split()[1:]
            print cnt,get_recognized_entity(i.split(),is_result_empty = False)[1:]
            cnt += 1
    '''
    #input = ['嗜碱性粒细胞绝对值','6','10^9/L','1.8--6.3']
    #input = ['中性粒细胞百分率','m1WT1','50.76','40-75','1']
    #input = ['嗜碱性粒细胞百分率','BU4S0{0L52','0-1','%']
    #print get_recognized_entity(input,is_result_empty = False)
    pass





