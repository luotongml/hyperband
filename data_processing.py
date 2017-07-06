
# coding: utf-8

from __future__ import division

import os
import numpy as np
import pandas as pd
import datetime
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

def save_options(df, path):
    print("saving data to {}".format(path))
    pd.to_pickle(df, path)

def to_sample_options(path="data/sample_option.pkl"):
    df = load_options()
    pd.to_pickle(df[:10],path)

def extract_option_features(df):
    df['moneyness'] = df['mid']/df['exeprice']
    #df = df[['mid','years_to_exe', 'exedate', 'etf_mid', 'exeprice']].dropna()
    df = df[['mid', 'years_to_exe', 'moneyness','etf_mid']].dropna()
    df_y = df['mid']
    df_x = df[['years_to_exe', 'moneyness','etf_mid']]
    return (df_x, df_y)


def train_test_split(df, window = 2):
    date = np.unique(df.index.date)

    for i in range(len(date)-window):
        train = df.loc[(df.index > datetime.datetime.combine(date[i],datetime.time.min)) & (df.index < datetime.datetime.combine(date[i+window],datetime.time.min))]
        test = df.loc[(df.index >= datetime.datetime.combine(date[i+window],datetime.time.min)) & (df.index <= datetime.datetime.combine(date[i+window],datetime.time.max))]
        (x_train,y_train) = extract_option_features(train)
        (x_test, y_test) = extract_option_features(test)
        data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
        yield data
