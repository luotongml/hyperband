
# coding: utf-8

# In[2]:

from __future__ import division

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vollib.black_scholes.implied_volatility import implied_volatility
from vollib.black_scholes.greeks.numerical import delta, theta, vega, gamma

from statsmodels.tsa.stattools import adfuller

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

#get_ipython().magic('matplotlib notebook')


# In[3]:

infos = pd.read_csv('../data/info.csv')
infos.columns = [i[11:] for i in infos.columns]
infos = infos[['tradedate', 'securityid', 'callorput', 'exeprice', 'exedate']].copy()
infos['securityid'] = infos['securityid'].astype(str)
infos['exedate'] = infos['exedate'].astype(str)
infos['tradedate'] = infos['tradedate'].astype(str)
infos['callorput'] = infos['callorput'].str.lower()
infos=infos.drop_duplicates()

available = os.listdir('../data/securities/')
available = [i for i in available if i !='sh510050.csv']
available = [i[2:-4] for i in available]

to_choose = list(set(infos['securityid'].unique()) & set(available))

infos = infos[infos['securityid'].apply(lambda x: x in available)]


# In[4]:

etf = pd.read_csv('../data/securities/sh510050.csv', parse_dates={'sectick.time_stamp':['sectick.tradedate', 'sectick.time']})
etf.columns = [i[8:] for i in etf.columns]
etf.set_index('time_stamp', inplace=True)
etf = etf.between_time('09:30:01', '14:56:59')
etf['mid'] = (etf['buyprice1'] + etf['sellprice1']) / 2
etf = etf[['mid', 'buyprice1', 'sellprice1']]
etf = etf.resample('1min', label='right', closed='right').last()
etf.dropna(inplace=True)
etf.columns = ['etf_' + i for i in etf.columns]


# In[5]:

def read_option(key):
    try:
        option = pd.read_csv('../data/securities/sh' + key + '.csv', parse_dates={'sectick.time_stamp':['sectick.tradedate', 'sectick.time']})
        option.columns = [i[8:] for i in option.columns]
    except:
        option = pd.read_csv('../data/securities/sh' + key + '.csv', parse_dates={'time_stamp':['tradedate', 'time']})
    option.set_index('time_stamp', inplace=True)
    option = option[(option['buyvolume1']>0) & (option['sellvolume1']>0)]
    
    
    option = option.between_time('09:30:01', '14:56:59')
#     option_am = option.between_time('09:33:00','09:35:00')
#     option_pm = option.between_time('14:51:00', '14:54:00')
#     option = pd.concat([option_am, option_pm])
    option.sort_index(inplace=True)
    
    
    option['mid'] = (option['buyprice1']+option['sellprice1']) / 2
    option = option[['securityid', 'mid', 'buyprice1', 'sellprice1']]
    option['securityid'] = option['securityid'].str[2:]
    option = option.resample('1min', label='right', closed='right').last()
    option.dropna(inplace=True)
    return option


# In[6]:

calls = infos[infos['callorput']=='c']


# In[7]:

calls[calls['tradedate'] == '20160112']


# In[8]:

train_dates = ['20160112', '20160113']
test_dates = ['20160114']


# In[9]:

all_dates = train_dates + test_dates

# In[33]:

all_data = []
for date in all_dates:
    this_call = calls[calls['tradedate'] == date]
    assert this_call['securityid'].duplicated().any() == False
    for i, row in this_call.iterrows():
        option = read_option(row['securityid'])
        option['exedate'] = pd.to_datetime(row['exedate'] + ' 14:57:00')
        option['minutes_to_exe'] = (option['exedate'] - option.index) / np.timedelta64(1,'m')
        option['years_to_exe'] = option['minutes_to_exe'] / (365*24*60)
        option['etf_mid'] = etf['etf_mid']
        option['exeprice'] = row['exeprice']
        option = option.loc[row['tradedate']]
        all_data.append(option)


# In[34]:

big_df = pd.concat(all_data)


# In[67]:

big_df.to_pickle('./bigdf.pkl')


# In[11]:

big_df = pd.read_pickle('./bigdf.pkl')


# In[12]:

big_df


# In[13]:

from sklearn.ensemble import GradientBoostingRegressor


# In[14]:

train = []
for d in train_dates:
    t = big_df.loc[d]
    train.append(t)


# In[15]:

test = []
for d in test_dates:
    t = big_df.loc[d]
    test.append(t)


# In[16]:

train = pd.concat(train)
test = pd.concat(test)


# In[23]:

train = train.dropna()
test = test.dropna()


# In[24]:

train_y = train['mid']
train_x = train[['years_to_exe', 'etf_mid', 'exeprice']]

test_y = test['mid']
test_x = test[['years_to_exe', 'etf_mid', 'exeprice']]


# In[25]:

alpha = 0.95


# In[26]:

gbr = GradientBoostingRegressor(n_estimators=300, random_state=0, alpha=alpha, loss='quantile')


# In[27]:

gbr.fit(train_x, train_y)


# In[ ]:

upper = gbr.


# In[ ]:




# In[41]:

xgb.fit(train_x, train_y)


# In[42]:

xgb.score(train_x, train_y)


# In[43]:

xgb.score(test_x, test_y)


# In[44]:

train


# In[45]:

test


# In[47]:

fig,ax=plt.subplots()
ax.scatter(xgb.predict(test_x), test_y)


# In[49]:

test['predict_mid'] = xgb.predict(test_x)


# In[51]:

secs = test['securityid'].unique()


# In[52]:

secs


# In[64]:

train['predict_mid'] = xgb.predict(train_x)


# In[65]:

dt = train[train['securityid']=='10000507']


# In[66]:

fig,ax=plt.subplots()
ax.plot(dt['mid'])
ax.plot(dt['predict_mid'])


# In[ ]:



