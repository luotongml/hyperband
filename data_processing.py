
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

def to_sample_options(path="data/sample_option.pkl"):
    df = load_options()
    pd.to_pickle(df[:10],path)

def extract_option_features(df):
    df = df['mid','years_to_exe', 'exedate', 'etf_mid', 'exeprice'].dropna()
    df_y = df['mid']
    df_x = df[['years_to_exe', 'etf_mid', 'exeprice']]
    return (df_x, df_y)


def train_test_split(big_df, window = 2):
    date = np.unique(big_df.index.date)
    day1 = datetime.timedelta(days=1)

    for i in range(len(date)-window):
        #TODO:loc doesnt seem to work with date selection
        train = big_df.loc[date[i].strftime('%Y-%m-%d'):date[i+window].strftime('%Y-%m-%d')]
        test = big_df.loc[date[i + window].strftime('%Y-%m-%d'):(date[i + window]+day1).strftime('%Y-%m-%d')]

        (x_train,y_train) = extract_option_features(train)
        (x_test, y_test) = extract_option_features(test)

        data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

        yield data
