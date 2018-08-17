import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings
import os
os.chdir(r'H:\数据挖掘mode')
warnings.filterwarnings('ignore')
data = pd.read_excel('aviation.xlsx',index_col='MEMBER_NO')
data = pd.DataFrame(data)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
le = OneHotEncoder()
# del data['MEMBER_NO']
for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = le.fit_transform(data[i])


data = data.sample(5000)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.4,random_state=1)
# 使用决策树
clf = DecisionTreeClassifier(max_depth=10,max_features=6)    # 初始化逻辑回归分类器
# 拟合模型
clf.fit(train_x,train_y)
test_x['per_y'] = clf.predict(test_x)

num = test_x[test_x['per_y']==1]
#前多少个值加起来大等0.9就取几个
# for i in range(10):
#     pc = PCA(n_components=i)     # 初始化PCA方法
#     nu_pca = pc.fit_transform(num)    # 拟合num数据生成新nu_pca主成分特征表
#     print(pc.explained_variance_ratio_)   # 打印主成分的方差解释比
pc = PCA(n_components=3)
nu_pca = pc.fit_transform(num)
print(nu_pca)
#确定聚类个数
# k_number = []
# for i in range(6,15):
#     km = KMeans(n_clusters=i)
#     km.fit(nu_pca)
#     k_number.append(km.inertia_)#添加离差平方和到建立的列表内
# plt.plot(range(6,15),k_number)
# plt.show()#看拐点的斜率 来决定要聚类的个数
km = KMeans(n_clusters=6)
km.fit(nu_pca)
#创建一个新的字段作为类标签
num['cluster'] = km.labels_
#获取要求字段
num = num[['cluster','FFP_days','DAYS_FROM_LAST_TO_END',
           'Points_Sum','L1Y_Flight_Count','avg_discount']]
num = pd.DataFrame(num)
#按标签分组 求分组后每一列的平均
num = num.groupby(by='cluster').mean()
#排序
num.sort_values(by=['Points_Sum',
                    'avg_discount',
                    'L1Y_Flight_Count',
                    'FFP_days',
                    'DAYS_FROM_LAST_TO_END'],
                ascending=[False,False,False,False,True],inplace=True)
#生成一个Series 来表示用户的价值
data = pd.DataFrame({'客户类型':['重要发展','重要价值','一般发展','一般保持','一般发展','一般挽留']})
num['客户类型'] = data#作为一个字段赋给num
num.to_csv('666.csv')#导出








