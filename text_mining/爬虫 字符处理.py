#调用获取网页信息的库
from urllib import request
#筛选网页信息
from bs4 import BeautifulSoup
from jieba import analyse
import pandas as pd
import jieba
from wordcloud import WordCloud#词云
import matplotlib.pyplot as plt
from gensim import models,corpora
import re
#写入要获取的网页的网址 给它编码不然会出问题
html1 = request.urlopen('http://www.biquge.info/8_8705/6740638.html') \
.read().decode('utf-8')
#创建一个txt文件 每次覆盖 编码中文
file = open('AAAAAA.txt','w',encoding='utf-8')
#放入此方法可以调用一些筛选方法
html = BeautifulSoup(html1,'html.parser')
#获取div中 id 为content中的文本
txt = str(html.select('div #content'))
txt = txt.replace('<br/><br/>','\n')
file.write(str(txt))
r = '[A-Za-z0-9]+'

word = ''
with open('AAAAAA.txt',encoding='utf-8') as f:
    for i in f.readlines():
        i.strip()
        i = re.sub(r,'',i)
        word += i
word = word.replace('[<div id="content">','')
word = word.replace('</div>]','')
stop = ''
with open('stopwords.txt',encoding='utf-8') as f:
    for i in f.readlines():
        stop +=i
        stop += '\xa0'
word = jieba.lcut(word)
word = filter(lambda x:x not in stop,word)
word = list(filter(lambda x:len(x)>1,word))
txt = word
word = pd.Series(word).map(lambda x:[x])
dict_word = corpora.Dictionary(word)
corpus = [dict_word.doc2bow(i) for i in word]
mode = models.LdaModel(corpus,num_topics=5,id2word=dict_word)
for i in range(5):
    print(mode.print_topics()[i])



# top10 = analyse.extract_tags(txt,10,withWeight=True)
# for i in range(10):
#     print(top10[i])
colud = WordCloud(font_path='search-ms:displayname=“Windows%20(C%3A)”中的搜索结果&crumb=location:C%3A%5C\Fonts\STXINWEI.TTF')
word_c = colud.generate_from_text(str(txt))
plt.imshow(word_c)
plt.axis('off')
plt.show()
