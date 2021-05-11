# find the efficient factors as much as possible
import pandas as pd
import talib
from talib.abstract import *
import datetime
from datetime import datetime
import math


# all num 1 express long market while all 0 represent short market.
def macd_feature_01(dif,dea):
    if dif > 0 and dea > 0:
        return 1
    else:
        return 0


def add_macd_factor(df):
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['hist_diff'] = df['macd_hist'].diff(periods=1)
    df['macd_feature_01'] = [macd_feature_01(df['macd'][i], df['macd_signal'][i]) for i in range(len(df))]
    df['macd_feature_02'] = list(map(lambda x: int(x>0), df['macd_hist']))
    df['macd_feature_03'] = list(map(lambda x: int(x>0), df['hist_diff']))

    return df


def add_macd_cross_factor(df):
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26,
                                                                    signalperiod=9)

    df = df.dropna(axis=0, how='any')
    signal_ls = [0]

    for i in range(df.index[0]+1, df.index[-1]+1):

        if df['macd'][i] > df['macd_signal'][i] and df['macd_signal'][i-1] > df['macd'][i-1]:
            signal_ls.append(1)
        else:
            signal_ls.append(0)

    assert (len(df) == len(signal_ls))

    df['macd_cross_up_signal'] = signal_ls

    signal_ls2 = [0]
    for i in range(df.index[0]+1, df.index[-1]+1):
        if df['macd'][i] < df['macd_signal'][i] and df['macd_signal'][i - 1] < df['macd'][i - 1]:
            signal_ls2.append(1)
        else:
            signal_ls2.append(0)
    assert (len(df) == len(signal_ls2))
    df['macd_cross_down_signal'] = signal_ls2

    return df[1:]


def add_kd_factor(df):
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=36,
                                           slowk_period=9, slowk_matype=0, slowd_period=9, slowd_matype=0)
    df['kd_feature_01'] = list(map(lambda x, y: int((x-y) > 0), df['slowk'], df['slowd']))
    df['kd_feature_02'] = list(map(lambda x: int(x > 80), df['slowd']))
    df['kd_feature_03'] = list(map(lambda x: int(x < 20), df['slowd']))

    return df


def add_ma_factor(df):
    df['ema_50'] = talib.EMA(df['close'], timeperiod = 50)
    df['ema_200'] = talib.EMA(df['close'], timeperiod = 200)
    df['sma_60'] = talib.SMA(df['close'], timeperiod = 60)
    df['sma_120'] = talib.SMA(df['close'], timeperiod=120)
    df['ma_feature_01'] = list(map(lambda x, y: int((x-y) > 0), df['ema_50'], df['ema_200']))
    df['ma_feature_02'] = list(map(lambda x, y: int((x-y) > 0), df['sma_60'], df['sma_120']))

    return df


def add_ma_cross_factor(df, m, n): # m < n : m cross n
    df['ema_'+str(m)] = talib.EMA(df['close'], timeperiod=m)
    df['ema_'+str(n)] = talib.EMA(df['close'], timeperiod=n)
    df = df.dropna(axis=0, how='any')

    signal_ls = [0]
    for i in range(df.index[0]+1, df.index[-1]+1):
        if df['ema_'+str(m)][i] > df['ema_'+str(n)][i] and df['ema_'+str(m)][i-1] < df['ema_'+str(n)][i-1]:
            signal_ls.append(1)
        else:
            signal_ls.append(0)
    assert (len(df) == len(signal_ls))
    df['ema_'+str(m)+'_cross_'+str(n)] = signal_ls

    return df[1:]


def add_rsi_factor(df):
    df['rsi_14'] = talib.RSI(df['close'], timeperiod = 14)
    df['rsi_5'] = talib.RSI(df['close'], timeperiod = 5)
    df['rsi_feature_01'] = list(map(lambda x: int(x > 80), df['rsi_14']))
    df['rsi_feature_02'] = list(map(lambda x: int(x < 20), df['rsi_14']))
    df['rsi_feature_03'] = list(map(lambda x, y: int((x - y) > 0), df['rsi_5'], df['rsi_14']))

    return df


def add_roc_factor(df, n):
    df['roc_'+str(n)] = talib.ROC(df['close'], timeperiod = n)
    df['rocp_'+str(n)] = talib.ROCP(df['close'], timeperiod = n)
    df = df.dropna(axis=0, how='any')

    target_y = df['rocp_'+str(n)].tolist()[n:]
    df = df[:-n]
    df['yp'+str(n)] = target_y
    df['y'+str(n)] = list(map(lambda x: int(x > 0), target_y))

    return df


# Below are indicators about volatility
def add_atr_factor(df):
    df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod = 14)
    df['atr_ma'] = talib.SMA(df['atr_14'], timeperiod = 5)
    df['atr_feature_01'] = df['atr_ma'].diff(periods=1)

    return df


def chebyshev_ineq_factor(ave_v, devup, close_price):
    std_var = (devup - ave_v) / 3
    if close_price > ave_v:
        k = (close_price - ave_v) / std_var
        if k < 1:
            up_probability = 0.5
        else:
            up_probability = 0.5 / (k*k)
    else:
        k = (ave_v - close_price) / std_var
        if k < 1:
            up_probability = 0.5
        else:
            up_probability = 1 - 0.5 / (k*k)
    return up_probability


# data processing method for continuous value in case of decision tree algorithm
def bi_partition_coding_chebyshev_factor(x):
    if x > 0.9:
        return 1
    elif x > 0.7:
        return 2
    elif x > 0.5:
        return 3
    elif x > 0.3:
        return 4
    elif x > 0.1:
        return 5
    else:
        return 6


def add_bbands_factor(df):
    df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(df['close'], timeperiod = 30, nbdevup = 3, nbdevdn = 3, matype = 0)
    df['bbands_feature_01'] = [chebyshev_ineq_factor(df['middleband'][i], df['upperband'][i], df['close'][i]) for i in range(len(df))]
    df['bbands_feature_02'] = [bi_partition_coding_chebyshev_factor(df['bbands_feature_01'][i]) for i in range(len(df))]

    return df


# time-related factors shown below
def add_time_factor(df):
    df = df.rename(columns={'Unnamed: 0': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['week'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month

    return df


