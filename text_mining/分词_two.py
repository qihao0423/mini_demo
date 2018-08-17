import pandas as pd
import jieba
from sklearn import model_selection
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import  ensemble
from sklearn import metrics
import xgboost
data = pd.read_excel('classify.xlsx',sheetname='classify')
data = pd.DataFrame(data)
num_post = data['post_type'].count()
data.drop(['User_Name','post_type','IP'],axis=1,inplace=True)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data = data[data['sentiment'] != 0]
data['Body'] = data['Body'].map(lambda x:jieba.lcut(x))
stop = ''

with open('stopwords.txt',encoding='utf-8') as f:
    for i in f.readlines():
        i.strip()
        stop += '\r\n P '
        stop+=i
data['Body'] = data['Body'].map(lambda x:[i for i in x if i not in stop])
def r(word):
    list = []
    for i in word:
        r = '[A-Za-z0-9]+'
        s = re.sub(r,'',i)
        list.append(s)
    return list
data['Body'] = data['Body'].map(lambda x:r(x))
data['Body'] = data['Body'].map(lambda x:[i for i in x if len(i)>1])
data = data[['Body','sentiment']]
data['Body'] = data['Body'].map(lambda x:' '.join(i for i in x))
str = data['Body'].tolist()
vect = CountVectorizer()
x = vect.fit_transform(str)
y = data['sentiment']
X_train, X_test, y_train, y_test = model_selection.train_test_split(x,y,train_size=0.6,random_state=1)
bys = BernoulliNB()
xgb = xgboost.XGBClassifier()
cv1 = model_selection.cross_val_score(bys, x, y, cv=5, scoring='accuracy')
cv2 = model_selection.cross_val_score(xgb, x, y, cv=5, scoring='accuracy')
print(cv1.mean())
print(cv2.mean())



