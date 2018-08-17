import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
#数据
data = pd.read_csv("ex2data1.txt")#读取数据
data.columns = (['x1','x2','y'])#给每列加个表名
data = data.sort_values(['x1','x2'])#根据x1为主x2为次排序
x1 = data['x1'].as_matrix().reshape(-1,1) #as_matrix()不加这个是一列数  加上是列向量
x2 = data['x2'].as_matrix().reshape(-1,1)
x = np.c_[x1,x2]
y = data['y'].as_matrix()
m = x1.shape[0]
logistic = LogisticRegression()
logistic.fit(x,y)       #x 是x1 x2左右合并 y 是列向量
print(logistic.intercept_)
print(logistic.coef_)
a = np.c_[3,4]
print(logistic.predict(a))
#画出两种的散点图
for i in range(m):
    if(y[i] == 1):
        plt.plot(x[i,0],x[i,1],'rx')
    if (y[i] == 0):
        plt.plot(x[i,0], x[i,1],'y*')
x1 = data['x1'].as_matrix().reshape(-1,1)
#根据x1计算出x2   就是模型的x值为0时 y值是0.5 theta[0]+theta[1]*x1+theta[2]*x2 = 0
x2 = -(logistic.intercept_+logistic.coef_[0][0]*x1)/logistic.coef_[0][1]
print(logistic.coef_[0][0])
plt.plot(x1,x2)
plt.show()

