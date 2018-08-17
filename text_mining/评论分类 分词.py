import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost
#负面
neg = pd.read_csv('meidi_jd_neg.txt',header=None,encoding='utf-8')
neg.rename(columns={0:'title'},inplace=True)
neg['lable'] = 1
#正面
pos = pd.read_csv('meidi_jd_pos.txt',header=None,encoding='utf-8')
pos.rename(columns={0:'title'},inplace=True)
pos['lable'] = 0
#根据列字段合并
data = pd.concat([pos,neg],ignore_index=True)
data['title'] = data['title'].map(lambda x:jieba.lcut(x))
#读取停用词
stop = ''
with open('stopwords.txt',encoding='utf-8') as f:
    for i in f.readlines():
        i.strip('\n')
        stop +=i
#过滤停用词
data['title'] = data['title'].map(lambda x:[i for i in x if i not in stop])
#上面这句代码的语法是：列表推导式子。意思是说，如果i不在停用词列表(stop)中，就保留该词语（也就是最前面的一个i），否则就进行删除
#上面的这句代码中，把for i in x看做整体，把if i not in stop看做判断语句，把最前面的i看做满足if语句之后的执行语句即可。
#单字过滤
def f(word):
    #filter 前面是true 后面就返回一个列表，false就不返回  用做过滤
    word = filter(lambda x:len(x) > 1, word)
    return list(word)
data['title'] = data['title'].map(f)
#将每一行从列表格式转化为字符串格式
data['title'] = data['title'].map(lambda x:' '.join(i for i in x))
#一维Series转为列表
str = data['title'].tolist()
vect = CountVectorizer()
x = vect.fit_transform(str)
y = data.loc[:,'lable']
trainx,testx,trainy,testy = train_test_split(x,y,test_size=0.2,random_state=11)
model = xgboost.XGBClassifier()
model.fit(trainx,trainy)
print('准确率',metrics.accuracy_score(testy,model.predict(testx)))
#贝叶斯分类器
model_ = BernoulliNB()
model_.fit(trainx,trainy)
print('准确率',metrics.accuracy_score(testy,model_.predict(testx)))