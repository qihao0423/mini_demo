
#调用支持向量机
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x = np.loadtxt("x.txt",delimiter=',')
y = np.loadtxt('y.txt')-1
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.6)
svc = SVC(C=1)#实例化 里面有很多参数 可以点开看
svc.fit(X_train,y_train)
print(accuracy_score(y_test,svc.predict(X_test))*100,"%")