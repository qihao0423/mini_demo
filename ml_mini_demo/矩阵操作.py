from numpy import *
import matplotlib.pyplot as plt
x1=array([1.5,0.8,2.6,1.0,0.6,2.8,1.2,0.9,0.4,1.3,1.2,2.0,1.6,1.8,2.2])
y2=array([3.1,1.9,4.2,2.3,1.6,4.9,2.8,2.1,1.4,2.4,2.4,3.8,3.0,3.4,4.0])
x = c_[ones(x1.shape).T,x1.T]
y = c_[y2]
plt.scatter(x[:,1],y,c="y",marker="*")
plt.show()
def computCost(x,y,theta=[[0],[0]]):
    J = (1.0/2*(y.size))*sum(square(x.dot(theta)-y))
    return  J
def gradientDescent(x,y,theta=[[0],[0]],alpha=0.01,iter=1000):
    history = zeros(iter)
    for i in range(iter):
        history[i] = computCost(x,y,theta)
        deltaTheta = (1.0/y.size)*(x.T.dot(x.dot(theta)-y))
        theta = theta - alpha*deltaTheta

    return deltaTheta,history


theta,Cost_J = gradientDescent(x,y)
print(Cost_J)
plt.plot(Cost_J,"rx")
plt.xlim(0,200)
plt.show()

a = arange(0,5)
b = theta[0]+a*theta[1]
print(theta)
plt.scatter(x[:,1],y,s=30,marker='x',c='r',linewidths=1)
plt.plot(a,b)
plt.show()

# plt.scatter(x,y,c="y",marker="*")
# plt.show()

