
# coding: utf-8

# In[2]:

from __future__ import division

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from vollib.black_scholes.implied_volatility import implied_volatility
#from vollib.black_scholes.greeks.numerical import delta, theta, vega, gamma

from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_absolute_error, mean_squared_error



# In[3]:
path = "data/"

train_dates = ['20160112', '20160113']
test_dates = ['20160114']


# In[9]:

all_dates = train_dates + test_dates


big_df = pd.read_pickle(path+ 'bigdf.pkl')


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



gbr = GradientBoostingRegressor(n_estimators=300, random_state=0, alpha=alpha, loss='quantile')


# In[27]:

gbr.fit(train_x, train_y)



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



