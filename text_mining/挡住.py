import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
data = pd.DataFrame(data)
# print(data)
data_t = pd.pivot_table(data=data,values='KWH',index='CONS_NO',columns='DATA_DATE',fill_value=np.nan)
print(data_t)