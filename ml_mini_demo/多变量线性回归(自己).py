import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#加载数据
data = np.loadtxt('ex1data2.txt',delimiter=',')#有俩个特征
m = data.shape[0]#求第一维度的长度
xx = data[:,:2]#切割出俩个特征
y = data[:,-1].reshape(m,1)#切割出实际值 并且变为m行1列的矩阵
#特征缩放
xx_ave = np.mean(xx,0)#按xx的每一列求均值
sigma = np.std(xx,0,ddof=1)#按xx的每一列求标准差
xx -= np.tile(xx_ave,(m,1))#把一列俩行的平均值 x方向不变 y方向复制m行  并且每一个值都-=
xx /= np.tile(sigma,(m,1))#把一列俩行的平均值 x方向不变 y方向复制m行  并且每一个值都/=

x = np.hstack((np.ones((m,1)),xx))#生成一个m行1列的列向量并且按行合并
#代价函数
def computCost(x,y,theta):
    J = (1.0 / 2*m  )  *np.sum(np.square(x.dot(theta)-y))#代价函数
    return J
#梯度下降
def gradientdescent(x,y,theta,a=0.01,iter=1000):
    history = np.eye(iter)      #定一个 iter 行 1 列的列向量
    for i in range(iter):
        history[i] = computCost(x,y,theta)          #把每次的代价函数结果存入
        delta = (1.0/m)*(x.T.dot(x.dot(theta)-y))      # 每次调整量
        theta -=a*delta      #梯度下降
    return history,theta       #把俩个结果返回
theta1 = np.ones((3,1))      #初始化一个theta
history,theta = gradientdescent(x,y,theta1)      #调用函数并且传参
plt.plot(history)
plt.show()
per_y = x.dot(theta)
plt.plot(per_y,y,'ro')
plt.show()
a1 = np.arange(-1,1,step=0.01)      #生成-1 - 1 步长为0.01的列向量
a2 = np.arange(-1,1,step=0.01)

b = theta[0]+theta[1]*a1+theta[2]*a2      #直线
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(a1,a2,b,c='y',marker='*')
ax.scatter(x[:,1],x[:,2],y,c='r')
ax.set_xlim3d(-10,10)
ax.set_ylim3d(-10,10)
ax.set_zlim3d(200000,400000)
plt.show()