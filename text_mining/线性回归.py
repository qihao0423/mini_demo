import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV
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
liner = LinearRegression()
tree = DecisionTreeRegressor()
a = cross_val_score(liner,X_train,y_train,cv=5)
b = cross_val_score(tree,X_train,y_train,cv=5)
print('线性回归模型',np.mean(a))
# print('决策树回归模型',np.mean(b))
# grid = GridSearchCV(tree,param_grid={'max_depth':range(5,20,2),'min_samples_leaf':range(5,50,5)},cv=5)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# print(grid.best_score_)
lr = LinearRegression()
lr.fit(X_train,y_train)
print(lr.intercept_)
print(lr.coef_)
print('---------------------------------------')
print("均方误差",metrics.mean_squared_error(y_test,lr.predict(X_test)))
print("R",metrics.r2_score(y_test,lr.predict(X_test)))
