import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import GridSearchCV,train_test_split
import warnings
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.svm import SVC
warnings.filterwarnings('ignore')#去警告！！
os.chdir(r'H:\数据挖掘mode')#设置路径
data = pd.read_csv('donated.csv')
data = pd.DataFrame(data)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=12345)
#定义三个模型
tree = DecisionTreeClassifier()
svc = SVC()
rfor = ensemble.RandomForestClassifier()
#投票集成学习   estimator内放入的是列表内元祖 元祖第一个是字符串 第二个是模型
voti = ensemble.VotingClassifier(estimators=[('决策树',tree),('支持向量机',svc),('随机森林',rfor)])
label_ = ['tree','SVM', 'Random Forest', 'Ensemble']
clf_ = [tree,svc,rfor,voti]
#循环显示评估值
for clf,lab in zip(clf_,label_):
    scores = model_selection.cross_val_score(clf,x,y,scoring='accuracy')
    #平均值 和 二倍标准差的关系
    print("accuracy: %0.2f (+/- %0.2f)[%s]" % (scores.mean(), scores.std() * 2, lab))




























