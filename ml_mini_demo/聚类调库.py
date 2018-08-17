#调用KMeans聚类库
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
data = np.array([[2,5],[4,6],[3,1],[6,4],[7,2],[8,4],[2,3],[3,1],[5,7],[6,9],[12,16],[10,11],[15,19],[16,12],[11,15],[10,14],[19,11],[17,14],[16,11],[13,19]])
km = KMeans(n_clusters=2)#里面参数是设置聚类数量 内还有其他参数点开看！
km.fit(data)#拟合
per_y = km.predict(data)#预测分类标签的值 如果是俩类就是0 1 三类就是 0 1 2
print(per_y)
center_ju = km.cluster_centers_#分类中心点的值在一个二维列表里
juli = km.inertia_#聚类中心均值点的总和
for i in range(len(data)):
    if(per_y[i] == 0):#如果预测标签为 0 类
        plt.plot(data[i,0],data[i,1],'rx')
    if (per_y[i] == 1):#如果预测标签为 1 类
        plt.plot(data[i, 0], data[i, 1],'y*')
#显示中心点
plt.plot(center_ju[0][0],center_ju[0][1],'bx')
plt.plot(center_ju[1][0],center_ju[1][1],'g*')
plt.show()
print(juli)