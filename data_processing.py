
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


#steps
# load all data and merge as necessary
# extract features x and y variables
# normal train/test split, [train_x, train_y], [test_x, test_y]
# rolling window train/test split generator of [train_x, train_y] [ test_x, test_y]



def load_options(path="data/bigdf.pkl"):
    print("loading data {} ....".format(path))
    df = pd.read_pickle(path)
    return df


def extract_option_features(df):
    df_y = df['mid']
    df_x = df[['years_to_exe', 'etf_mid', 'exeprice']]
    return (df_x, df_y)

def tran_test_split((df_x, df_y), rolling=False):
    train_dates = ['20160112', '20160113']
    test_dates = ['20160114']
    big_df = []

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



    test_y = test['mid']
    test_x = test[['years_to_exe', 'etf_mid', 'exeprice']]

    data = {'x_train': train_x, 'y_train': train_y, 'x_test': test_x, 'y_test': test_y}
    return data
