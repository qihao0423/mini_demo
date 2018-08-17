# "已知数据X1为学生期末考试分数（满分100），X2为学生平时表现（满分5），Y为学生是否可以顺利升班（1为升班，0为末班）
# X1=[60,61,67,99,98,78,80,85,81,94,74,88,84,89,79]
# X2=[2,2.3,2.5,4,3.5,2.1,5,4.6,3.1,4.2,5,4.3,2,1.8,4.5]
# Y=[0,0,0,1,1,0,1,1,0,1,1,1,0,0,1]
# 1.画出0-1分布图
# 2.实现正则化逻辑回归"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X1=np.array([60,61,67,99,98,78,80,85,81,94,74,88,84,89,79]).reshape(-1,1)
X2=np.array([2,2.3,2.5,4,3.5,2.1,5,4.6,3.1,4.2,5,4.3,2,1.8,4.5]).reshape(-1,1)
y=np.array([0,0,0,1,1,0,1,1,0,1,1,1,0,0,1])
m = X1.shape[0]
x = np.c_[X1,X2]
logisticRegression =LogisticRegression()
logisticRegression.penalty = "l2"   #选择正则化类型
logisticRegression.solver = "lbfgs" #l2 中四种中一种
logisticRegression.fit(x,y)
theta0 = logisticRegression.intercept_
theta1 = logisticRegression.coef_[0][0]
theta2 = logisticRegression.coef_[0][1]
for i in range(m):
    if(y[i] == 1):
        plt.plot(x[i,0],x[i,1],'rx')
    if (y[i] == 0):
        plt.plot(x[i, 0], x[i, 1], 'y*')

x1 = x[:,0]
x2 = -(theta0+x1*theta1)/theta2
plt.plot(x1,x2)
plt.show()
print(theta0,theta1,theta2)


#
# def gradient_descent(x,y,theta,a=0.01,iter=1500,lamda=1):
#     for i in range(iter):
#         h = 1/(1+np.exp(-(x.dot(theta))))
#         error = h -y
#         delta = x.T.dot(error)
#         theta -= a*(delta+(lamda/m)*(np.c_[[0],theta[1],theta[2]]).reshape(3,1))
#     return theta
# theta = np.ones((3,1))
# x = np.c_[np.ones((m,1)),x]
# standardScaler = StandardScaler()
# x= standardScaler.fit_transform(x)
# for i in range(m):
#     if(y[i] == 1):
#         plt.plot(x[i,1],x[i,2],'rx')
#     if (y[i] == 0):
#         plt.plot(x[i,1], x[i,2],'y*')
# theta = gradient_descent(x,y,theta)
# print(theta)
# x1 = x[:,1]
# x2 = -(theta[0]+theta[1]*x1)/theta[2]
# plt.plot(x1,x2)
# plt.show()


