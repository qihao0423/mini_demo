import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#实例化 点开看参数  现在写的参数是表示每层有多少个神经元
#逗号隔开再写是再加一层100个的神经元
mLPClassifier = MLPClassifier(hidden_layer_sizes=(100,100))
data = np.loadtxt("xigua.txt",delimiter=',')
x = data[:,:-1]
y = data[:,-1]
mLPClassifier.fit(x,y)#拟合
pred_y = mLPClassifier.predict(x)#得到预测结果
#accuracy_score(y,pred_y)运用此方法计算出准确率
print(accuracy_score(y,pred_y))
print(y)
print(pred_y)
