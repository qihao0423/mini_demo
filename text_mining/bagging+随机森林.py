import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import GridSearchCV,train_test_split
import warnings
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.svm import SVC
warnings.filterwarnings('ignore')
os.chdir(r'H:\数据挖掘mode')
data = pd.read_csv('donated.csv')
data = pd.DataFrame(data)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=12345)
bc = ensemble.BaggingClassifier(base_estimator=SVC())
'''
    base_estimator=None        基分类器---默认是决策树
    n_estimators=10            分类器数量
    max_samples=1.0            最大样本数
    max_features=1.0           最大特征数
    bootstrap=True             随机取样本是否放回
    bootstrap_features=False   随机取特征是否放回
    n_jobs=1                   并行化
    random_state=None          随机状态
'''
parm = {'n_estimators':range(10,30,10),
        'max_samples':np.arange(0.1,1.0,0.3),
        'max_features': range(1,5,1),           #这个好像就得给小点
        }
grid = GridSearchCV(bc,parm,scoring='f1',cv=5)
grid.fit(x,y)
print(grid.best_params_)
print(grid.best_score_)

#------------------------------------------------------------# rf = RandomForestClassifier()
rf = ensemble.RandomForestClassifier()
'''
    n_estimators=10              分类器数量
    criterion="gini"             分裂准则
    max_depth=None               最大数层级
    min_samples_split=2          最小分割样本数
    min_samples_leaf=1,          最小剪枝样本数
    max_features="auto"         划分时考虑的特征数
    max_leaf_nodes=None         最大叶子节点数
    n_jobs=1                    并行化
    random_state=None           随机状态
    class_weight=None           类别权重
'''
# cv2 = cross_val_score(rf,x,y,scoring='roc_auc',cv=10)
# print(cv2.mean())
rf.fit(x,y)
result = pd.DataFrame({'feature':data.columns[0:-1],
                       'importance':rf.feature_importances_})
#输出根据 特征重要度高低排序的重要特征
print(result.sort_values(by='importance',ascending=False))
