import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('ftx.xlsx')
data = pd.DataFrame(data)

a = data.query("年==2009&月 in (4,5,6) & 销售区域 in ('北京','上海','广州')").groupby(by='销售区域')['销售数量'].sum()
a = pd.DataFrame(a)
a['销售数量'] = a
a['销售区域'] = ['北京','上海','广州']
data['月'] = pd.cut(data['月'],[1,4,7,10,13],right=False,labels=['第一季度','第二季度','第三季度','第四季度'])
print(data)
zui = data.groupby(by=['年','月'])['销售数量'].sum()
print(zui)
plt.bar(range(len(zui)),zui)
plt.show()