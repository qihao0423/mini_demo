import pandas as pd
import jieba as jieba
from jieba import analyse
from jieba import posseg
import gensim
import warnings
warnings.filterwarnings('ignore')
txt = ''
with open('背影.txt',encoding='gbk') as f:
    for i in f.readlines():
        i.split()
        txt += i
#得到前二十个关键词
top20 = analyse.extract_tags(txt,20,withWeight=True)
for i in range(20):
    print(top20[i])
print('-------------------')
w = jieba.lcut(txt)
stop = ''
with open('stoplist.txt',encoding='utf-8') as f:
    for i in f.readlines():
        stop +=i

'''filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判，
然后返回 True 或 False，最后将返回 True 的元素放到新列表中'''
word = filter(lambda x:len(x)>1,w)
word = list(filter(lambda x:x not in stop,word))
#打印词性  内参数必须放字符串
cixing = posseg.lcut(str(word))
print(cixing)
#转为为字典前的格式好像必须是 列表内的列表 如 [[1],[2]]
word = pd.Series(word).map(lambda x:[x])
#生成语料
dict = gensim.corpora.Dictionary(word)
#可以输出映射关系
# print(dict.token2id)
#将词表转化为词袋  前一个是索引 后一个是次数
words = [dict.doc2bow(i) for i in word]
#第一个词袋 第二个主题个数 第三个语料
lda = gensim.models.LdaModel(words,num_topics=5,id2word=dict)
#打印每一个主题
for i in lda.print_topics():
    print(i)
