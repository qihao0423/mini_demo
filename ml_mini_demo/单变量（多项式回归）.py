import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


data = np.loadtxt("ex1data1.txt",delimiter=',')
x = data[:,0].reshape(-1,1)
y = data[:,-1].reshape(-1,1)
linearRegression = LinearRegression()
pol = PolynomialFeatures(4)  #设置提高次幂的程度  4就是最高4次  前面依次 3,2,1
pol_x = pol.fit_transform(x) #构成一个新的特征矩阵 第一列1 2列是本身1次 3列2次 4列3次
linearRegression.fit(pol_x,y)#调用库拟合数据模型
pre_y = linearRegression.predict(pol_x)#求成预测值
#生成一个该特征最大到最小的列向量 把步长设置的小一点
#这样求预测结果时可以尽可能多的求出预测值 因为比较密集所以可以连成一条线
xian_x = np.arange(min(x),max(x),0.1).reshape(-1,1)
#把生成的数据也 变为多项式所需的矩阵形势
xian_xx = pol.fit_transform(xian_x)
#求出自己生成数据的预测值
xian_y = linearRegression.predict(xian_xx)
#这就是线
plt.plot(xian_x,xian_y,'b')
#这是原数据分布
plt.plot(x,y,'r*')
plt.show()