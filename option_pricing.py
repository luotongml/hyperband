
# coding: utf-8

# In[33]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

get_ipython().magic('matplotlib notebook')


# In[34]:

data = pd.read_pickle('/home/dev/jupyterhub/assignments/hbli/data/options_midprice_predict_data.pkl')


# 查看期权基本信息

# In[35]:

data[['securityid', 'callorput', 'exeprice', 'exedate',]].drop_duplicates()


# 提取建模数据及resample

# In[36]:

data_call = data[data['callorput']=='c']
data_raw = data_call.groupby('securityid').resample('1D').last()
data_raw = data_raw.dropna()


# In[37]:

date = data_raw['tradedate'].dt.strftime('%Y-%m-%d').unique()


# In[38]:

i=100
window = 3


# In[39]:

# 选出训练集和测试集
date_train = date[slice(i, i+window)]
data_train = data_raw.loc[(slice(None), date_train), ['midprice', 'S_K', 'years_to_exe']]
data_train = data_train[data_train['years_to_exe'] > 0]

date_test = date[i+window]
data_test = data_raw.loc[(slice(None), date_test), :]

X_train = data_train[['S_K', 'years_to_exe']]
y_train = data_train['midprice']

X_test = data_test[['S_K', 'years_to_exe']]
y_test = data_test['midprice']


# 标准化
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)


# In[ ]:




# 神经网络建模 Neural Networks

# In[40]:

regr_nn = MLPRegressor(hidden_layer_sizes=(10, 8), activation='relu', solver='adam', )
regr_nn.fit(X_train_std, y_train)

Rsquare_nn = regr_nn.score(X_train_std, y_train)
mse_nn = mean_squared_error(y_test.values, y_test_predict_nn)
y_train_predict_nn = regr_nn.predict(X_train_std)

y_test_predict_nn = regr_nn.predict(X_test_std)


# In[ ]:




# 随机森林建模 Radom Forest

# In[41]:

regr_rf = RandomForestRegressor(n_estimators=20, criterion='mse', max_depth=30, )
regr_rf.fit(X_train_std, y_train)

Rsquare_rf = regr_rf.score(X_train_std, y_train)
mse_rf = mean_squared_error(y_test.values, y_test_predict_rf)
y_train_predict_rf = regr_rf.predict(X_train_std)

y_test_predict_rf = regr_rf.predict(X_test_std)


# In[ ]:




# 决策树DecisionTree 和 AdaBoost DecisionTree建模

# In[42]:

# Fit regression model
regr_dt = DecisionTreeRegressor(criterion='mse', max_depth=4)
regr_abdt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300,)

regr_dt.fit(X_train_std, y_train)
regr_abdt.fit(X_train_std, y_train)

Rsquare_dt = regr_dt.score(X_train_std, y_train)
mse_dt = mean_squared_error(y_test.values, y_test_predict_dt)
y_train_predict_dt = regr_dt.predict(X_train_std)

Rsquare_abdt = regr_abdt.score(X_train_std, y_train)
mse_abdt = mean_squared_error(y_test.values, y_test_predict_abdt)
y_train_predict_abdt = regr_abdt.predict(X_train_std)

y_test_predict_dt = regr_dt.predict(X_test_std)
y_test_predict_abdt = regr_abdt.predict(X_test_std)


# In[ ]:




# boosting建模

# In[43]:

regr_gb = GradientBoostingRegressor(n_estimators=100, max_depth=1, loss='ls')

regr_gb.fit(X_train_std, y_train)

Rsquare_gb = regr_gb.score(X_train_std, y_train)
mse_gb = mean_squared_error(y_test.values, y_test_predict_gb)

y_train_predict_gb = regr_gb.predict(X_train_std)
y_test_predict_gb = regr_gb.predict(X_test_std)


# In[ ]:




# 合并训练集与测试集的结果

# In[48]:

for model in ['nn', 'rf', 'dt', 'abdt', 'gb']:
	y_train_predict = 'y_train_predict_'+model
	y_train_predict = eval(y_train_predict)
	y_train_predict = pd.Series(y_train_predict)
	y_train_predict.index = y_train.index
	
	result_train = pd.DataFrame(y_train)
	result_train['predict_'+model] = y_train_predict
	result_train['error_'+model] = result_train['midprice'] - result_train['predict_'+model]
	
	mse = 'mse_'+model
	mse= eval(mse)
	print 'MSE_'+model+' : ' + str(mse)
	
	Rsquare = 'Rsquare_'+model
	Rsquare = eval(Rsquare)
	print 'Rsquare_'+model+' : ' + str(Rsquare)
	print 'train error_'+model+' mean: ' + str(result_train['error_'+model].mean())


	y_test_predict = 'y_test_predict_'+model
	y_test_predict = eval(y_test_predict)
	y_test_predict = pd.Series(y_test_predict)
	y_test_predict.index = y_test.index

	result_test = pd.DataFrame(y_test)
	result_test['predict_'+model] = y_test_predict
	result_test['error_'+model] = result_test['midprice'] - result_test['predict_'+model]
	print 'test error_'+model+' mean: ' + str(result_test['error_'+model].mean())
	print('------------')

