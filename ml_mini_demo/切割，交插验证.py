import numpy as np
#调用神经网络
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#第一个切割训练测试集 第二个交叉验证
from sklearn.model_selection import train_test_split,cross_val_score
x = np.loadtxt("x.txt",delimiter=',')
y = np.loadtxt('y.txt')
#train_size=0.6  训练集占总比例的0.6切割
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.6)
mlp = MLPClassifier()#实例化
#【参数依次 模型 ， 特征 ， 实际值 ，    交插验证组数】
cc = cross_val_score(mlp,x_train,y_train,cv=10)
print(cc.mean())#如上的话  交插验证10组的结果的平均值