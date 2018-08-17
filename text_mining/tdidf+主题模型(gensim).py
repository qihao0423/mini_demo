
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(r'H:\数据挖掘mode')
infile = 'xyj.txt'
import re
import jieba
import pandas as pd
from jieba import analyse as ans
from gensim import corpora,models
#读取待处理文本
r = u'[？。，；：“”！（）]?'
li = ''
with open(infile,'r',encoding='gbk',errors='ignore') as f:
    for i in f.readlines():
        i= i.strip('\n')
        li += re.sub(r,'',i)
words = jieba.lcut(li)
# print(words)
#其中sentence为待提取的文本，topK为返回几个TF/IDF权重最大的关键词，
# 默认值为20。withWeight=True 同时显示每个词的tfidf
top = ans.extract_tags(li,topK=30,withWeight=True)
for i in top:
    print(i)
print('---------------------------------------------------')
#读取停用词
stop = ''
with open('stopwords.txt','r',encoding='gbk',errors='ignore') as s:
    for line in s:
        line = line.strip()
        stop += line
# 停用词过滤
word = []
for i in words:
    if len(i) >1:
        if i not in stop:
            word.append(i)
    else:
        pass
# 将列表转换为数组
word = pd.Series(word)
word = word.map(lambda x:[x])
print(word)
print('----------------------------------------')
# 建立词典 这里需要放入的好像是列表格式
s = corpora.Dictionary(word)
#如果想要查看单词与编号之间的映射关系：
print(s.token2id)
print('------------------------------------------------------------')
# 为了真正将记号化的文档转换为向量，需要：
#函数doc2bow()简单地对每个不同单词的出现次数进行了计数，并将单词转换为其编号，
# 然后以稀疏向量的形式返回结果。因此，稀疏向量[(0, 1), (1, 1)]表示：
# 在“Human computer interaction”中“computer”(id 0) 和“human”(id 1)各出现一次；
# 其他10个dictionary中的单词没有出现过（隐含的）
corpus = [s.doc2bow(i) for i in word]
# print(corpus)
#通过TFIDF向量生成LDA模型，id2word表示编号的对应词典，num_topics表示主题数
slda = models.LdaModel(corpus,num_topics=5,id2word=s)

print(slda.print_topics(num_words=3,num_topics=3))


