import jieba
from jieba import analyse
import re
from gensim import models,corpora
from jieba import posseg
import pandas
import wordcloud
import matplotlib.pyplot as plt
r = '[A-Za-z0-9]+'
str = ''
with open('AAAAAA.txt',encoding='utf-8') as a:
    for line in a.readlines():
        line.strip()
        line = re.sub(r,'',line)
        str += line
stop = ''
with open('stopwords.txt',encoding='utf-8') as f:
    for i in f.readlines():
        stop +=i
        stop += '\xa0'
jieba.add_word('全国统')
text = jieba.lcut(str)
text = filter(lambda x:x not in stop,text)
text = list(filter(lambda x:x not in stop,text))
text = list(filter(lambda x:len(x)>1,text))

text = pandas.Series(text).map(lambda x:[x])

dict_word = corpora.Dictionary(text)
corpus = [dict_word.doc2bow(i) for i in text]

# mode = models.LdaModel(corpus=corpus,num_topics=3,id2word=dict_word)
#
# for i in range(3):
#     print(mode.print_topics()[i])
# text_top1 = analyse.textrank(str,topK=5,withWeight=True,allowPOS=('n'))
# print(text_top1)


# cloud = wordcloud.WordCloud(font_path='search-ms:displayname=“Windows%20(C%3A)”中的搜索结果&crumb=location:C%3A%5C\Fonts\STXINWEI.TTF',
#                 background_color='black',
#                 width=700,
#                 height=300,
#                 max_words=2000,
#                 max_font_size=40)
# print(str(text))
# word_c = cloud.generate_from_text(str(text))
# plt.imshow(word_c)
# plt.show()

poss = jieba.posseg.lcut(str)
print(poss)