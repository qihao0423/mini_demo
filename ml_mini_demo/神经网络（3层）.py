from numpy import *;
import numpy as np;

X = np.array([[1,0.05,0.1]]).T
y = np.array([[0.01,0.99]]).T
#random是0 - 1 之间
theta_1 = 2*np.random.random((2,3)) - 1#每一层的theta 以 后一层作为行这一行作为它的列
theta_2 = 2*np.random.random((2,3)) - 1
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x) #此处X代表的是S  函数本身！！！(1 + np.exp(-x))的导数
    return 1 / (1 + np.exp(-x))
for j in range(1000):
    #正向传播
    a_2 = nonlin(np.dot(theta_1,X))  #第二层的输出，预测
    a_2 = vstack(([1],a_2))#偏置一个  1
    a_3 = nonlin(np.dot(theta_2,a_2)) #第三层的输出值
    #反向传播
    delta_3 = (a_3 - y)*nonlin(a_3,True) #实际上就在以高 确信度减小预测误差。后半部分有点争议
    #这一行的误差 = 这一行的theta的转置 * 下一行的误差 * 这一行值的S型函数的导数
    delta_2 = theta_2.T.dot(delta_3) * nonlin(a_2,True) #实际上就在以高确信度减小预测误差。
    delta_2=delta_2[0:2]#切剩下一个还是俩个  并无多大影响

    theta_2 -= delta_3.dot(a_2.T) #更新第2层到底3层的权重
    theta_1 -= delta_2.dot(X.T) #更新第1层到底2层的权重
print(a_3)
print(y)