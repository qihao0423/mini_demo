import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_excel('违约.xlsx')
data = pd.DataFrame(data)
ps_mean = data['PS_OD_DT'].mean()
data['PS_OD_DT'] = pd.cut(data['PS_OD_DT'],bins=[min(data['PS_OD_DT']),ps_mean,max(data['PS_OD_DT'])],right=False,labels=[0,1])

for i in data.columns:
    if data[i].isnull().sum() / len(data[i]) > 0.7:
        data.drop(i,axis=1,inplace=True)


data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

one = LabelEncoder()
for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = one.fit_transform(data[i])
data = data.sample(2000)
y = data.iloc[:,0]
x = data.iloc[:,1:]


# tree = DecisionTreeClassifier()
# forst = ensemble.RandomForestClassifier()
# def computing(model,scoring):
#
#     cv1 = model_selection.cross_val_score(model,x,y,scoring=scoring,cv=2)
#     cv2 = model_selection.cross_val_score(model,x,y,scoring=scoring,cv=2)
#     print(cv1.mean())
#     print(cv2.mean())
# computing(tree,'f1')
# computing(forst,'roc_auc')

X_train, X_test, y_train, y_test = model_selection.train_test_split(x,y,train_size=0.6,random_state=1)
ranforst = ensemble.RandomForestClassifier(n_estimators=100)
ranforst.fit(X_train,y_train)
per_y = ranforst.predict(X_test)

# print(metrics.confusion_matrix(y_test,per_y))
print(metrics.precision_score(y_test,per_y))
print(metrics.recall_score(y_test,per_y))
proba = ranforst.predict_proba(X_test)
a,b,c = metrics.roc_curve(y_test,proba[:,1])
plt.plot(a,b)
plt.show()
