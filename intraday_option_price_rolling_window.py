
# coding: utf-8

# In[1]:

from __future__ import division

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
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

import scipy.stats as stats

get_ipython().magic('matplotlib notebook')


# In[2]:

data = pd.read_pickle('/home/dev/jupyterhub/assignments/hbli/data/options_midprice_predict_data.pkl')


# In[3]:

data


# 提取建模数据及resample

# In[4]:

data_call = data[data['callorput']=='c']
data_call['time_stamp'] = data_call.index

data_call_raw = data_call.groupby('securityid').resample('1min').last()
data_call_raw = data_call_raw.dropna()


# 对call类型期权，判断虚实值的状态

# In[5]:

def intravalue(x):
    if x > 0.1:
        return 'DITM'
    elif 0.05< x <= 0.1:
        return 'ITM'
    elif -0.05 <= x <= 0.05:
        return 'ATM'
    elif -0.1 <= x < -0.05:
        return 'OTM'
    elif x < -0.1:
        return 'DOTM'


# In[8]:

data_call_raw['intrasive_value'] = data_call_raw['etf_midprice'] - data_call_raw['exeprice']
data_call_raw['intravalue_status'] = data_call_raw['intrasive_value'].map(intravalue)


# In[9]:

data_call_raw[['etf_midprice', 'exeprice', 'intrasive_value', 'intravalue_status']]


# 对实虚值状态的期权进行筛选

# In[11]:

#data_raw = data_call_raw[(data_call_raw['intravalue_status'] == 'ATM')]
#data_raw = data_call_raw[(data_call_raw['intravalue_status'] == 'ITM') | (data_call_raw['intravalue_status'] == 'DITM')]
#data_raw = data_call_raw[(data_call_raw['intravalue_status'] != 'DOTM')] # 导致删除后的数据时间不连续

data_raw = data_call_raw


# In[12]:

data_raw


# rolling Window 多种模型建模预测

# In[13]:

date = data_raw['tradedate'].dt.strftime('%Y-%m-%d').unique()
window = 2


# In[14]:

delete = pd.DataFrame()
evaluate = pd.DataFrame()
result = pd.DataFrame()
for i in range(len(date)-window):
    # train and test set
    date_train = date[slice(i, i+window)]
    data_train = pd.DataFrame()
    for j in range(window):
        data_train_temp = data_raw.loc[(slice(None), date_train[j]), ['midprice', 'S_K', 'years_to_exe', 'exedate', 'exeprice', 'etf_midprice', 'intravalue_status']]
        data_train_temp = data_train_temp.loc[data_train_temp['exedate'] != date_train[j]]  # 删除行权日当天的数据
        data_train = data_train.append(data_train_temp)

    date_test = date[i+window]
    data_test = data_raw.loc[(slice(None), date_test), ['midprice', 'S_K', 'years_to_exe', 'exedate', 'exeprice', 'etf_midprice', 'intravalue_status']]
    data_test = data_test.loc[data_test['exedate'] != date_test]
    data_test_row0 = data_test.shape[0]
    print date_test+' the test set rows: '+str(data_test_row0)
    # test data in the range
    S_min = data_train['etf_midprice'].min()
    S_max = data_train['etf_midprice'].max()
    data_test = data_test.loc[(S_min < data_test['etf_midprice']) & (data_test['etf_midprice'] < S_max)]
    data_test_row1 = data_test.shape[0]
    print date_test+' the test set in the train range rows: '+str(data_test_row1)
    print date_test+' the test set delete rows: '+str(data_test_row0 - data_test_row1)
    print '-----------'

    # 只选择test中moneyness在某一类的数据进行预测
    #data_test = data_test[data_test['intravalue_status'] != 'DOTM']
    
    # 统计强假设下的删除率
    test_delete = {'date':date_test, 'test_rows_raw':data_test_row0, 'test_rows_delete':data_test_row0 - data_test_row1}
    test_delete = pd.DataFrame([test_delete])
    delete = delete.append(test_delete)

    if data_test_row1 == 0:
    	continue

    X_train = data_train[['S_K', 'years_to_exe']]
    y_train = data_train['midprice']
    X_test = data_test[['S_K', 'years_to_exe']]
    y_test = data_test['midprice']
    # standardize
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    # model
    # Neural Networks
    regr_nn = MLPRegressor(hidden_layer_sizes=(10, 8), activation='relu', solver='adam', )
    regr_nn.fit(X_train_std, y_train)
    Rsquare_nn = regr_nn.score(X_train_std, y_train)
    y_train_predict_nn = regr_nn.predict(X_train_std)
    y_test_predict_nn = regr_nn.predict(X_test_std)
    train_mse_nn = mean_squared_error(y_train.values, y_train_predict_nn)
    train_mae_nn = mean_absolute_error(y_train.values, y_train_predict_nn)
    test_mse_nn = mean_squared_error(y_test.values, y_test_predict_nn)
    test_mae_nn = mean_absolute_error(y_test.values, y_test_predict_nn)
    # Random Forest
    regr_rf = RandomForestRegressor(n_estimators=30, criterion='mse', max_depth=10, )
    regr_rf.fit(X_train_std, y_train)
    Rsquare_rf = regr_rf.score(X_train_std, y_train)
    y_train_predict_rf = regr_rf.predict(X_train_std)
    y_test_predict_rf = regr_rf.predict(X_test_std)
    train_mse_rf = mean_squared_error(y_train.values, y_train_predict_rf)
    train_mae_rf = mean_absolute_error(y_train.values, y_train_predict_rf)
    test_mse_rf = mean_squared_error(y_test.values, y_test_predict_rf)
    test_mae_rf = mean_absolute_error(y_test.values, y_test_predict_rf)
    # Decision Tree & AdaBoost Decision Tree
    regr_dt = DecisionTreeRegressor(criterion='mse', max_depth=8)
    regr_abdt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=300,)
    regr_dt.fit(X_train_std, y_train)
    regr_abdt.fit(X_train_std, y_train)
    Rsquare_dt = regr_dt.score(X_train_std, y_train)
    y_train_predict_dt = regr_dt.predict(X_train_std)
    Rsquare_abdt = regr_abdt.score(X_train_std, y_train)
    y_train_predict_abdt = regr_abdt.predict(X_train_std)
    y_test_predict_dt = regr_dt.predict(X_test_std)
    y_test_predict_abdt = regr_abdt.predict(X_test_std)
    train_mse_dt = mean_squared_error(y_train.values, y_train_predict_dt)
    train_mae_dt = mean_absolute_error(y_train.values, y_train_predict_dt)
    test_mse_dt = mean_squared_error(y_test.values, y_test_predict_dt)
    test_mae_dt = mean_absolute_error(y_test.values, y_test_predict_dt)
    train_mse_abdt = mean_squared_error(y_train.values, y_train_predict_abdt)
    train_mae_abdt = mean_absolute_error(y_train.values, y_train_predict_abdt)
    test_mse_abdt = mean_squared_error(y_test.values, y_test_predict_abdt)
    test_mae_abdt = mean_absolute_error(y_test.values, y_test_predict_abdt)
    # Boosting
    regr_gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, loss='ls')
    regr_gb.fit(X_train_std, y_train)
    Rsquare_gb = regr_gb.score(X_train_std, y_train)
    y_train_predict_gb = regr_gb.predict(X_train_std)
    y_test_predict_gb = regr_gb.predict(X_test_std)
    train_mse_gb = mean_squared_error(y_train.values, y_train_predict_gb)
    train_mae_gb = mean_absolute_error(y_train.values, y_train_predict_gb)
    test_mse_gb = mean_squared_error(y_test.values, y_test_predict_gb)
    test_mae_gb = mean_absolute_error(y_test.values, y_test_predict_gb)
    # XG Boosting
    regr_xgb = XGBRegressor(n_estimators=300, max_depth=3, objective='reg:linear')
    regr_xgb.fit(X_train_std, y_train)
    Rsquare_xgb = regr_xgb.score(X_train_std, y_train)
    y_train_predict_xgb = regr_xgb.predict(X_train_std)
    y_test_predict_xgb = regr_xgb.predict(X_test_std)
    train_mse_xgb = mean_squared_error(y_train.values, y_train_predict_xgb)
    train_mae_xgb = mean_absolute_error(y_train.values, y_train_predict_xgb)
    test_mse_xgb = mean_squared_error(y_test.values, y_test_predict_xgb)
    test_mae_xgb = mean_absolute_error(y_test.values, y_test_predict_xgb)


    evaluate_1day = {'testdate':date_test}
    result_train = pd.DataFrame()
    result_test = pd.DataFrame()
    # 合并结果
    for model in ['nn', 'rf', 'dt', 'abdt', 'gb', 'xgb']:
        y_train_predict = 'y_train_predict_'+model
        y_train_predict = eval(y_train_predict)
        y_train_predict = pd.Series(y_train_predict)
        y_train_predict.index = y_train.index
        
        if result_train.empty:
            result_train = pd.DataFrame(y_train)
            result_train['predict_'+model] = y_train_predict
            result_train['error_'+model] = result_train['midprice'] - result_train['predict_'+model]
        else:
            result_train['predict_'+model] = y_train_predict
            result_train['error_'+model] = result_train['midprice'] - result_train['predict_'+model]
    
        y_test_predict = 'y_test_predict_'+model
        y_test_predict = eval(y_test_predict)
        y_test_predict = pd.Series(y_test_predict)
        y_test_predict.index = y_test.index
    
        if result_test.empty:
            result_test = pd.DataFrame(y_test)
            result_test['intravalue_status'] = data_test['intravalue_status']
            result_test['predict_'+model] = y_test_predict
            result_test['error_'+model] = result_test['midprice'] - result_test['predict_'+model]
        else:
            result_test['predict_'+model] = y_test_predict
            result_test['error_'+model] = result_test['midprice'] - result_test['predict_'+model]
        
        trainmsename = 'train_mse_'+model
        trainmse= eval(trainmsename)
        trainmaename = 'train_mae_'+model
        trainmae= eval(trainmaename)
        Rsquarename = 'Rsquare_'+model
        Rsquare = eval(Rsquarename)
        testmsename = 'test_mse_'+model
        testmse= eval(testmsename)
        testmaename = 'test_mae_'+model
        testmae= eval(testmaename)
    
        evaluate_1model = {Rsquarename:Rsquare, trainmsename:trainmse, trainmaename:trainmae,                            testmsename:testmse, testmaename:testmae}
        # 合并多个模型的评估结果
        evaluate_1day = dict(evaluate_1day, **evaluate_1model)

    # 合并每个周期模型的评估结果
    evaluate_1day = pd.DataFrame([evaluate_1day])
    evaluate = evaluate.append(evaluate_1day)
    # 合并每个周期模型的预测结果
    result = result.append(result_test)


# In[16]:

# 合并期权买卖1档数据
result = result.merge(data_raw[['buyprice1', 'sellprice1']], left_index=True, right_index=True, how='left')


# In[17]:

result


# #### 保存结果
# result.to_pickle('option_price_predict_freq1min_window'+ str(window) +'day_result.pkl')
# evaluate.to_pickle('option_price_predict_freq1min_window'+ str(window) +'day_evaluate.pkl')
# delete.to_pickle('option_price_predict_freq1min_window'+ str(window) +'day_delete.pkl')

# ## 对预测集moneyness分类进行统计错误率

# In[19]:

result_error = result[['midprice', 'intravalue_status', 'error_nn', 'error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']]

# 选择某一类的moneyness期权
result_error_moneyness1 = result_error[result_error['intravalue_status'] == 'DITM']

#result_error_moneyness2 = result_error[(result_error['intravalue_status'] == 'ITM') | (result_error['intravalue_status'] == 'DITM')]
result_error_moneyness2 = result_error[(result_error['intravalue_status'] == 'ITM')]

result_error_moneyness3 = result_error[result_error['intravalue_status'] == 'ATM']

#result_error_moneyness4 = result_error[(result_error['intravalue_status'] == 'OTM') | (result_error['intravalue_status'] == 'DOTM')]
result_error_moneyness4 = result_error[(result_error['intravalue_status'] == 'OTM')]

result_error_moneyness5 = result_error[(result_error['intravalue_status'] == 'DOTM')]


print 'all test data rows: ' + str(result_error.shape[0])
print 'all test mae result: '
print result_error[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'DITM test data rows: ' + str(result_error_moneyness1.shape[0])
print 'DITM test mae result: '
print result_error_moneyness1[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'ITM test data rows: ' + str(result_error_moneyness2.shape[0])
print 'ITM test mae result: '
print result_error_moneyness2[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'ATM test data rows: ' + str(result_error_moneyness3.shape[0])
print 'ATM test mae result: '
print result_error_moneyness3[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'OTM test data rows: ' + str(result_error_moneyness4.shape[0])
print 'OTM test mae result: '
print result_error_moneyness4[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'DOTM test data rows: ' + str(result_error_moneyness5.shape[0])
print 'DOTM test mae result: '
print result_error_moneyness5[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()



# ## 计算误差与价格的比率

# In[20]:

result_error_ratio = result[['midprice', 'error_nn', 'error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].apply(lambda x: x/result['midprice'])
result_error_ratio['intravalue_status'] = result['intravalue_status'].copy()

# 选择某一类的moneyness期权
result_error_ratio_moneyness1 = result_error_ratio[result_error_ratio['intravalue_status'] == 'DITM']

#result_error_ratio_moneyness2 = result_error_ratio[(result_error_ratio['intravalue_status'] == 'ITM') | (result_error_ratio['intravalue_status'] == 'DITM')]
result_error_ratio_moneyness2 = result_error_ratio[(result_error_ratio['intravalue_status'] == 'ITM')]

result_error_ratio_moneyness3 = result_error_ratio[result_error_ratio['intravalue_status'] == 'ATM']

#result_error_ratio_moneyness4 = result_error_ratio[(result_error_ratio['intravalue_status'] == 'OTM') | (result_error_ratio['intravalue_status'] == 'DOTM')]
result_error_ratio_moneyness4 = result_error_ratio[(result_error_ratio['intravalue_status'] == 'OTM')]

result_error_ratio_moneyness5 = result_error_ratio[(result_error_ratio['intravalue_status'] == 'DOTM')]


print 'all test data rows: ' + str(result_error_ratio.shape[0])
print 'all test mae ratio: '
print result_error_ratio[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'DITM test data rows: ' + str(result_error_ratio_moneyness1.shape[0])
print 'DITM test mae ratio: '
print result_error_ratio_moneyness1[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'ITM test data rows: ' + str(result_error_ratio_moneyness2.shape[0])
print 'ITM test mae ratio: '
print result_error_ratio_moneyness2[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'ATM test data rows: ' + str(result_error_ratio_moneyness3.shape[0])
print 'ATM test mae ratio: '
print result_error_ratio_moneyness3[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'OTM test data rows: ' + str(result_error_ratio_moneyness4.shape[0])
print 'OTM test mae ratio: '
print result_error_ratio_moneyness4[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()
print '------------------------'
print 'DOTM test data rows: ' + str(result_error_ratio_moneyness5.shape[0])
print 'DOTM test mae ratio: '
print result_error_ratio_moneyness5[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].abs().mean()


# In[ ]:




# ## 查看数据删除率与估计误差

# In[21]:

# 查看数据删除率
#delete = pd.read_pickle('./option_price_predict_freq1min_window'+ str(window) +'day_delete.pkl')
delete.set_index('date', inplace=True)
delete['delete_ratio'] = delete['test_rows_delete'] / delete['test_rows_raw']


# In[22]:

delete.describe()


# In[23]:

# 查看估计误差
print evaluate.mean()


# 检验error是否显著为负

# In[24]:

#result = pd.read_pickle('./option_price_predict_freq1min_window2day_result.pkl')
result


# In[25]:

result_error = result[['error_rf', 'error_dt', 'error_abdt', 'error_gb', 'error_xgb']].reset_index().copy()

result_error_1option = result_error[result_error['securityid']=='10000646']
result_error_1option.set_index('time_stamp', inplace = True)
result_error_1option_1day = result_error_1option.loc['2016-07-05']


# In[26]:

result_error.describe()


# In[27]:

result_error_1option.describe()


# In[28]:

result_error_1option_1day.describe()


# In[29]:

stats.ttest_1samp(result_error['error_xgb'], 0, axis=0)


# In[ ]:




# In[ ]:




# 将预测数据与原始数据合并

# In[30]:

result_xgb = result[['predict_xgb']]
result_predict_xgb = result_xgb.merge(data_raw, left_index= True, right_index=True)


# In[31]:

result_predict_xgb


# 画出test中的真实值与预测值的散点图，看是否有预测值的异常点

# In[32]:

#result_predict_xgb[['midprice', 'predict_xgb']].plot.scatter('midprice', 'predict_xgb')


# 任意选出一支期权看存续期内估计值与真实值的走势

# In[33]:

#选择某一天数据
data_plot_1option = result_predict_xgb.loc[('10000646', slice(None)), ['etf_midprice', 'midprice', 'predict_xgb', 'intravalue_status']]
data_plot_1option = data_plot_1option.reset_index()
data_plot_1option.set_index('time_stamp', inplace=True)

fig,ax = plt.subplots()
ax.plot(data_plot_1option['etf_midprice'], 'y')
ax1 = ax.twinx()
ax1.plot(data_plot_1option['midprice'], 'b')
ax1.plot(data_plot_1option['predict_xgb'], 'r')


# In[34]:

plot_date = '2016-07-07'

#10000631-10000635期权的生命期为20160526-20160727


# In[35]:

#10000646的生命期为20160623-20160824，K=2.0，S变化范围2.054-2.237，越来越in the money
#10000646的生命期为20160623-20160824
#选择某一天数据
data_plot_1option_1day = result_predict_xgb.loc[('10000645', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
#data_plot_1option_1day = result_predict_xgb.loc[('10000631', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
data_plot_1option_1day = data_plot_1option_1day.reset_index()
data_plot_1option_1day.set_index('time_stamp', inplace=True)

fig,ax = plt.subplots()
ax.plot(data_plot_1option_1day['etf_midprice'], 'y')
ax1 = ax.twinx()
ax1.plot(data_plot_1option_1day['midprice'], 'b')
ax1.plot(data_plot_1option_1day['predict_xgb'], 'r')


# In[36]:

#10000646的生命期为20160623-20160824，K=2.05，S变化范围2.054-2.237，越来越in the money
#10000646的生命期为20160623-20160824
#选择某一天数据
data_plot_1option_1day = result_predict_xgb.loc[('10000646', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
#data_plot_1option_1day = result_predict_xgb.loc[('10000632', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
data_plot_1option_1day = data_plot_1option_1day.reset_index()
data_plot_1option_1day.set_index('time_stamp', inplace=True)

fig,ax = plt.subplots()
ax.plot(data_plot_1option_1day['etf_midprice'], 'y')
ax1 = ax.twinx()
ax1.plot(data_plot_1option_1day['midprice'], 'b')
ax1.plot(data_plot_1option_1day['predict_xgb'], 'r')


# In[37]:

#10000646的生命期为20160623-20160824，K=2.10，S变化范围2.054-2.237，越来越in the money
#10000646的生命期为20160623-20160824
#选择某一天数据
data_plot_1option_1day = result_predict_xgb.loc[('10000647', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
#data_plot_1option_1day = result_predict_xgb.loc[('10000633', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
data_plot_1option_1day = data_plot_1option_1day.reset_index()
data_plot_1option_1day.set_index('time_stamp', inplace=True)

fig,ax = plt.subplots()
ax.plot(data_plot_1option_1day['etf_midprice'], 'y')
ax1 = ax.twinx()
ax1.plot(data_plot_1option_1day['midprice'], 'b')
ax1.plot(data_plot_1option_1day['predict_xgb'], 'r')


# In[38]:

#10000646的生命期为20160623-20160824，K=2.15，S变化范围2.054-2.237，越来越in the money
#10000646的生命期为20160623-20160824
#选择某一天数据
data_plot_1option_1day = result_predict_xgb.loc[('10000648', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
#data_plot_1option_1day = result_predict_xgb.loc[('10000634', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
data_plot_1option_1day = data_plot_1option_1day.reset_index()
data_plot_1option_1day.set_index('time_stamp', inplace=True)

fig,ax = plt.subplots()
ax.plot(data_plot_1option_1day['etf_midprice'], 'y')
ax1 = ax.twinx()
ax1.plot(data_plot_1option_1day['midprice'], 'b')
ax1.plot(data_plot_1option_1day['predict_xgb'], 'r')


# In[ ]:




# In[39]:

#10000649的生命期为20160623-20160824，K=2.2，S变化范围2.054-2.237，到期前才达到in the money
#选择某一天数据
data_plot_1option_1day = result_predict_xgb.loc[('10000649', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
#data_plot_1option_1day = result_predict_xgb.loc[('10000635', plot_date), ['etf_midprice', 'midprice', 'predict_xgb']]
data_plot_1option_1day = data_plot_1option_1day.reset_index()
data_plot_1option_1day.set_index('time_stamp', inplace=True)

fig,ax = plt.subplots()
ax.plot(data_plot_1option_1day['etf_midprice'], 'y')
ax1 = ax.twinx()
ax1.plot(data_plot_1option_1day['midprice'], 'b')
ax1.plot(data_plot_1option_1day['predict_xgb'], 'r')

