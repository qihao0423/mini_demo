import pandas as pd

data = pd.read_excel('loans.xlsx')
data = pd.DataFrame(data)
data.drop('每月归还额',axis=1,inplace=True)
# print(data)
# print(data['贷款期限'].value_counts())
# print(data.groupby(by='还款状态')['贷款金额'].mean())
# print(data['贷款金额'].describe())
# print(data.sort_values(by=['发放贷款日期','贷款金额'],ascending=[False,True]))
# data['每月归还额'] = (data['贷款金额'])/(data['贷款期限'])
# print(data)
print(data[data['账户号']>2000][data['账户号']<5000][['发放贷款日期','贷款金额']])