import matplotlib.pyplot as plt
from sklearn import datasets
#降维库
from sklearn.decomposition import PCA
breast_cancer = datasets.load_breast_cancer()#加载数据
x = breast_cancer.data#获取特征
y = breast_cancer.target#分类两种数据列表
target_name = breast_cancer.target_names#俩种列表的名称
pac = PCA(n_components=2)#要下降到的目标维度
pac.fit(x)#拟合
x_new = pac.transform(x)#降维得到新特征
print("投射方差：",pac.explained_variance_ratio_)
color = ['red','yellow','blue']
lw = 2
#       ！！！！！！！！！目前懵比！！！！！！！
for color,i,target_name in zip(color,[0,1,2],target_name):
    plt.scatter(x_new[y == i, 0], x_new[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
    #图例
    plt.legend(loc='best', shadow=False, scatterpoints=1)#图例
    #标题
    plt.title('PCA of breast_cancer dataset')
plt.show()