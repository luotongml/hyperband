
# coding: utf-8

from __future__ import division

import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from vollib.black_scholes.implied_volatility import implied_volatility
#from vollib.black_scholes.greeks.numerical import delta, theta, vega, gamma
#from statsmodels.tsa.stattools import adfuller
#from sklearn.metrics import mean_absolute_error, mean_squared_error
#from sklearn.ensemble import GradientBoostingRegressor


def load_option(path="data/bigdf.pkl"):
    print("loading data {} ....".format(path))
    big_df = pd.read_pickle(path)
    train_dates = ['20160112', '20160113']
    test_dates = ['20160114']

    all_dates = train_dates + test_dates
    train = []
    for d in train_dates:
        t = big_df.loc[d]
        train.append(t)

    test = []
    for d in test_dates:
        t = big_df.loc[d]
        test.append(t)

    train = pd.concat(train)
    test = pd.concat(test)

    train = train.dropna()
    test = test.dropna()

    train_y = train['mid']
    train_x = train[['years_to_exe', 'etf_mid', 'exeprice']]

    test_y = test['mid']
    test_x = test[['years_to_exe', 'etf_mid', 'exeprice']]

    data = {'x_train': train_x, 'y_train': train_y, 'x_test': test_x, 'y_test': test_y}
    return data
