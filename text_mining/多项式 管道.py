import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder
from sklearn import pipeline
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn import metrics
from sklearn import ensemble
data = pd.read_csv('titanic.csv')
data = pd.DataFrame(data)
num_age = data['Age'].isnull().sum()
num_data = len(data)
print('%0.2f%%' % ((num_age/num_data)*100))
new_data = data[['Age','Sib_Sp','Parch','Fare']]
train_data = new_data[new_data.Age.isnull()]
aa = train_data.iloc[:,1:]
test_data = new_data[new_data.Age.notnull()]
a = test_data.iloc[:,1:]
b = test_data.iloc[:,0]
xx_train, xx_test, yy_train, yy_test = train_test_split(a,b,train_size=0.6,random_state=1)
clf1 = linear_model.Ridge()
clf1.fit(xx_train,yy_train)
print('\nR^2:',metrics.r2_score(yy_test,clf1.predict(xx_test)))

clf2 = linear_model.Lasso()
clf2.fit(xx_train,yy_train)
print('\nR^2:',metrics.r2_score(yy_test,clf2.predict(xx_test)))
clf3 = pipeline.Pipeline([('Poly',PolynomialFeatures()),
                          ('liner',linear_model.LinearRegression())
                          ])

clf3.fit(xx_train,yy_train)
print('\nR^2:',metrics.r2_score(yy_test,clf3.predict(xx_test)))
data.ix[data['Age'].isnull(),'Age'] = clf3.predict(aa)
data.dropna(inplace=True)
# 字符编码
for i in data.columns:
    if data[i].dtype == 'object':
        le = LabelEncoder()
        data[i] = le.fit_transform(data[i])
Sur = 'Survived'
lab = [i for i in data.columns if i not in Sur]
y = data.ix[:,Sur]
x = data.ix[:,lab]
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=1)
forst = ensemble.RandomForestClassifier()
gbdt = ensemble.GradientBoostingClassifier()
cv1 = cross_val_score(forst,X_train,y_train,cv=5,scoring='f1')
cv2 = cross_val_score(gbdt,X_train,y_train,cv=5,scoring='f1')
print(cv1.mean())
print(cv2.mean())
print('-------------------')
#
# grid = GridSearchCV(estimator=gbdt,param_grid={'learning_rate':np.arange(0.1,1,0.1),
#                                                'n_estimators':range(20,100,10),
#                                                'subsample':(0.7,0.8,0.1),
#                                                'max_depth':range(2,10,2)
#                                                },scoring='f1')
# grid.fit(X_train,y_train)
# print(grid.best_params_)
gbdt_new = ensemble.GradientBoostingClassifier(learning_rate=0.1,n_estimators=90,subsample=0.1,max_depth=4)
gbdt_new.fit(X_train,y_train)
print('\n准确率:',metrics.accuracy_score(y_test,gbdt_new.predict(X_test)))
