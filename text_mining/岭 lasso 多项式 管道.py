import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures#多项式处理
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline#管道
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'E:\AAAAAAAAAAAAAA每月笔记\小时训     人工智能\12.data mining\1511R\大数据_1511R（数据挖掘）_09_单元')
data = pd.read_excel('墨迹天气.xlsx')
data = pd.DataFrame(data)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
onehot = LabelEncoder()
for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = onehot.fit_transform(data[i])
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.6,random_state=1)
clf1 = LinearRegression()
clf1.fit(X_train,y_train)
'''R2方法是将预测值跟只使用均值的情况下相比，看能好多少。其区间通常在（0,1）之间。
0表示还不如什么都不预测，直接取均值的情况，而1表示所有预测跟真实结果完美匹配的情况。 
与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测
'''
print('均方误差',metrics.mean_squared_error(y_test,clf1.predict(X_test)))
print('R^2 ',int(round(metrics.r2_score(y_test,clf1.predict(X_test)),2)*100),'%')
#岭回归
ridge = linear_model.Ridge(alpha=10)
ridge.fit(X_train,y_train)
print('均方误差',metrics.mean_squared_error(y_test,ridge.predict(X_test)))
print('R^2 ',int(round(metrics.r2_score(y_test,ridge.predict(X_test)),2)*100),'%')
#LASSO回归
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
print('均方误差',metrics.mean_squared_error(y_test,lasso.predict(X_test)))
print('R^2 ',int(round(metrics.r2_score(y_test,lasso.predict(X_test)),2)*100),'%')
# 网格搜索
print('-------------岭回归--------------------')
gs1 = GridSearchCV(ridge,param_grid={'alpha':[0.01,0.1,1,10]},scoring='r2')
gs1.fit(X_train,y_train)
print(gs1.best_score_)
print(gs1.best_params_)
print('-----------LASSO回归-------------------')
gs2 = GridSearchCV(lasso,param_grid={'alpha':[0.01,0.1,1,10]},scoring='r2')
gs2.fit(X_train,y_train)
print(gs2.best_score_)
print(gs2.best_params_)
#多项式 管道 同时运行
model = Pipeline([('poly',PolynomialFeatures()),
                  ('liner',LinearRegression())])
#想要给上面管道里的模型设置参数就用 代表它的字符串 + __ 它的参数来表示！
#degree表示转化为多项式的程度！
#然后直接根据管道名运行就好了
for i in range(1,5,1):
    model.set_params(poly__degree = i)
    model.fit(X_train,y_train)
    print('均方误差', metrics.mean_squared_error(y_test,model.predict(X_test)))
    print('R^2 ', int(round(metrics.r2_score(y_test,model.predict(X_test)), 2) * 100), '%')



