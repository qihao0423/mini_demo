import numpy as np
import matplotlib.pyplot as plt

#缩放方法
def featureScaling(X):
    X -= np.mean(X,0)
    X /= np.std(X,0,ddof=1)
    X = np.c_[np.ones((m,1)),X]
    return X
#根据模型求出预测值
def comput_error(z):
    h = 1/(1+np.exp(-z))

    return h
#梯度下降
def gradientSecent(x,y,theta,a,iter):
    hhh = np.ones((m, 1))
    for i in range(iter):
        h = comput_error(x.dot(theta))#接受传回的误差
        hhh = h
        delta = x.T.dot(h-y)#代价函数
        theta -= a*delta#更新theta    梯度下降公式
    return hhh,theta
#计算模型准确率
def testModel(x,y,theta,m):
    flag = 0        #定义一个标记变量
    for i in range(m):
        h = comput_error(x[i].dot(theta)) #接受预测值
        if (np.where(h > 0.5,1,0) == y[i]):#如果任何h > 0.5 就为1 否则
            flag += 1                      #为 0 等于实际值  就加 1
    print(flag/m) #输出准确率
# 展示出 边界的线和散点图
def show_line(x,y,theta,m):
    for i in range(m):      #按照每行 每行
        if(y[i] == 1):      #判读是 红色x
            plt.plot(x[i,1],x[i,2],'rx')
        if (y[i] == 0):      #判读不是 黄色*
            plt.plot(x[i,1],x[i,2],'y*')
    x1 = x[:,1]
    x2 = -(theta[0] + x1*theta[1])/theta[2]#预测出不成立是的x2(自己这样想的)
    plt.plot(x1,x2)
    plt.show()

#数据
data = np.loadtxt("ex2data1.txt",delimiter=',')
m = data.shape[0]      #计算长度
x = data[:,:2]
y = data[:,-1].reshape(-1,1)
theta = np.ones((3,1))
x = featureScaling(x)      #去缩放 并且第一列弄成 1
hhh,theta = gradientSecent(x,y,theta,0.01,1500)
show_line(x,y,theta,m)      #展示散点图
testModel(x,y,theta,m)      #准确率






plt.plot(x.dot(theta),hhh,'rx')
plt.show()












