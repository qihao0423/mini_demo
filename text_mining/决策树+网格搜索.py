import pandas as pd
import numpy as np
#预处理库 将文字转化为标准化数字
from sklearn.preprocessing import LabelEncoder
#决策树分类
from sklearn.tree import DecisionTreeClassifier
#交插验证       网格搜索
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score
#index_col 是将某一列作为索引
data = pd.read_excel('airplane.xlsx',index_col='MEMBER_NO')
data = pd.DataFrame(data)
#删除缺失值  不删会出错
data.dropna(inplace=True)
#实例化
le = LabelEncoder()
#循环所有列
for i in data.columns:
    if data[i].dtype == 'object':#如果列名为字符串
        data[i] = le.fit_transform(data[i])#转换为数字形式
# print(data)
df = data.sample(5000)#随机取5000个数据
y = df.iloc[:,-1]
x = df.iloc[:,:-1]
tree = DecisionTreeClassifier(criterion='gini')
#交插验证 输出的是5个准确率 scoring后也可以写 'f1'
jiaocha = cross_val_score(tree,x,y,cv=5,scoring='accuracy')
# print(jiaocha)
#网格搜索   estimator是模型 param_grid是要找的最优参数（是字典格式）  scoring是评估最优大的依据
gr = GridSearchCV(estimator=tree,param_grid={'max_depth':range(5,20,2),'min_samples_leaf':range(5,50,5)},scoring='f1',cv=5)
#拟合模型
gr.fit(x,y)
#最好参数
print(gr.best_params_)
#最优评估值
print(gr.best_score_)
