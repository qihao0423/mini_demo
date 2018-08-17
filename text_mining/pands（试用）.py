import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_excel('aviation.xlsx')
data = pd.DataFrame(data)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data = data[data['runoff_flag']==1]
del data['MEMBER_NO']
data = data.sample(500)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
pca = PCA(n_components=3)
pca.fit(x)
# print(pca.explained_variance_ratio_)
new_x = pca.transform(x)
# a = []
# for i in range(1,10):
#     k = KMeans(n_clusters=i)
#     k.fit(new_x)
#     a.append(k.inertia_)
# plt.plot(range(1,10),a)
# plt.show()
k = KMeans(n_clusters=5)
k.fit(new_x)
x = x[['FFP_days','DAYS_FROM_LAST_TO_END',
           'Points_Sum','L1Y_Flight_Count','avg_discount']]
x['labels'] = k.labels_
x = x.groupby(by='labels').mean()
x = pd.DataFrame(x)
print(x)
x.sort_values(by=['FFP_days','DAYS_FROM_LAST_TO_END',
           'Points_Sum','L1Y_Flight_Count','avg_discount'],
              ascending=[False,True,False,False,True],inplace=True)
dangci = pd.DataFrame({'客户级别':['A','B','C','D','E']})
x['客户级别'] = dangci
x.to_csv(u'客户级别.csv')


