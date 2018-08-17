from wordcloud import WordCloud
import os
from os import path
import matplotlib.pyplot as plt
import jieba
from scipy.misc import imread


#读取图片
bg_pic = imread('H:\\数据挖掘mode\\QQ图片20180523170041.png')
comment_text=open('背影.txt','r').read()
cut_text=jieba.lcut(comment_text)
print(cut_text)
cut_text=filter(lambda x:len(x)>1,cut_text )
cut_text=list(cut_text)
cut_text=' '.join(cut_text)
print(cut_text)
#第一个参数 字体路径
cloud=WordCloud(font_path='search-ms:displayname=“Windows%20(C%3A)”中的搜索结果&crumb=location:C%3A%5C\Fonts\STXINWEI.TTF',
                background_color='black',
                width=700,
                height=300,
                max_words=2000,
                max_font_size=40,
                mask=bg_pic#该属性 字体显示形状 在非白色部分显示！！！
                )

#放入的是文本文件（必须是分词后的）  就是字符串
word_cloud=cloud.generate_from_text(cut_text)
plt.imshow(word_cloud)
plt.axis('off')#去掉X Y 轴坐标
plt.show()

