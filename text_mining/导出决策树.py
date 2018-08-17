import pandas as pd
import numpy as np
#导入决策树 后面是为了导出决策树的dot文档
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score,roc_curve
import matplotlib.pyplot as plt
#去警告
import warnings
warnings.filterwarnings('ignore')
import os
data = pd.read_excel('Mass.xlsx')
data = pd.DataFrame(data)

data.replace('?',np.nan,inplace=True)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
le = OneHotEncoder()
#转换编码 用的独热
for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = le.fit_transform(data[i])
print(data)
print(data.info())
tree = DecisionTreeClassifier()
grid = GridSearchCV(tree,param_grid={'max_depth':range(5,20,2),'min_samples_leaf':range(5,50,5)},cv=5,scoring='f1',)
grid.fit(x,y)
print(grid.best_params_)
print(grid.best_score_)
tree = DecisionTreeClassifier(max_depth=5,min_samples_leaf=30)
tree.fit_transform(x,y)
print("精确率为：",precision_score(y,tree.predict(x)))
a,b,c = roc_curve(y,tree.predict(x))
plt.plot(a,b)
plt.show()
#先是生成一个dot文档
export_graphviz(tree,'tree.dot')
#然后通过系统命令转化为pdf格式  暂且这样记
os.system('dot -Tpdf tree.dot -o trdd.pdf')