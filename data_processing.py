
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
    df = df['mid','years_to_exe', 'etf_mid', 'exeprice'].dropna()
    df_y = df['mid']
    df_x = df[['years_to_exe', 'etf_mid', 'exeprice']]
    return (df_x, df_y)


def train_test_split(big_df, window = 2):
    train_dates = ['20160112', '20160113']
    test_dates = ['20160114']

    date = big_df['tradedate'].dt.strftime('%Y-%m-%d').unique()

    for i in range(len(date)-window):
        train_dates = date[slice(i, i + window)]
        test_dates = date[i+window]

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

        (x_train,y_train) = extract_option_features(train)
        (x_test, y_test) = extract_option_features(test)

        data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

        return data