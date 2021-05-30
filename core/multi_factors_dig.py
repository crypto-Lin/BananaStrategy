# find the efficient factors as much as possible
import pandas as pd
import numpy as np
import talib
from talib.abstract import *
from core.utils import metric

import PyEMD
from PyEMD import EEMD

# all num 1 express long market while all 0 represent short market.
def macd_feature_01(dif, dea):
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


def add_ma_factor(df, m, n):
    df['ema_'+str(m)] = talib.EMA(df['close'], timeperiod=m)
    df['ema_'+str(n)] = talib.EMA(df['close'], timeperiod=n)
    df['sma_'+str(m)] = talib.SMA(df['close'], timeperiod=m)
    df['sma_'+str(n)] = talib.SMA(df['close'], timeperiod=n)
    df['ema_'+str(m)+'_'+str(n)] = list(map(lambda x, y: int((x-y) > 0), df['ema_'+str(m)], df['ema_'+str(n)]))
    df['sma_'+str(m)+'_'+str(n)] = list(map(lambda x, y: int((x-y) > 0), df['sma_'+str(m)], df['sma_'+str(n)]))

    return df


def add_ema_diff_factor(df, n, i):
    df['ema_'+str(n)] = talib.EMA(df['close'], timeperiod=n)
    df['diff_1'+str(n)] = df['ema_'+str(n)].diff(periods=i)
    df['diff1_'+str(n)] = list(map(lambda x: int(x > 0), df['diff_1'+str(n)]))
    df['diff_2'+str(n)] = df['ema_'+str(n)].diff(periods=i).diff(periods=i)
    df['diff2_'+str(n)] = list(map(lambda x: int(x > 0), df['diff_2'+str(n)]))

    return df


def add_ma_cross_factor(df, m, n): # m < n : m cross n
    df['ema_'+str(m)] = talib.EMA(df['close'], timeperiod=m)
    df['ema_'+str(n)] = talib.EMA(df['close'], timeperiod=n)
    df = df.dropna(axis=0, how='any')

    signal_ls = [0]
    for i in range(df.index[0]+1, df.index[-1]+1):
        if (df['ema_'+str(m)][i] > df['ema_'+str(n)][i]) and (df['ema_'+str(m)][i-1] < df['ema_'+str(n)][i-1]) :
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


# below factors are about simple pattern recognition
def add_pattern_reconition_factor(df):
    df['two_crows'] = talib.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
    df['three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
    df['three_inside'] = talib.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])
    df['three_line_strike'] = talib.CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close'])
    df['three_outside'] = talib.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close'])
    df['three_star_south'] = talib.CDL3STARINSOUTH(df['open'], df['high'], df['low'], df['close'])
    df['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
    df['abandoned_baby'] = talib.CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['close'])
    df['advance_block'] = talib.CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['close'])
    df['belt_hold'] = talib.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close'])
    df['break_away'] = talib.CDLBREAKAWAY(df['open'], df['high'], df['low'], df['close'])
    df['closing_marubozu'] = talib.CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['close'])
    df['conceal_baby_swall'] = talib.CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['close'])
    df['counter_attack'] = talib.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close'])
    df['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
    df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df['doji_star'] = talib.CDLDOJISTAR(df['open'], df['high'], df['low'], df['close'])
    df['dragon_fly_doji'] = talib.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close'])
    df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df['evening_doji_star'] = talib.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['gap_sideside_white'] = talib.CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['close'])
    df['grave_stone_doji'] = talib.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close'])
    df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['morning_doji_star'] = talib.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['on_neck'] = talib.CDLONNECK(df['open'], df['high'], df['low'], df['close'])

    return df


def add_down_pattern_recognition_factor(df):
    df['hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
    df['evening_doji_star'] = talib.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)
    df['three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
    df['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
    df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
    return df


def add_up_pattern_recognition_factor(df):
    df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['inverted_hammer'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) # not sure the direction
    df['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close']) # strong up trend
    df['morning_doji_star'] = talib.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)

    return df


def add_continuation_pattern_recognition_factor(df):
    df['three_method'] = talib.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close'])
    df['spin_top'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
    return df


def add_first_raising_limit_factor(df):
    signal_ls = [0]
    for i in range(df.index[0]+1, df.index[-1]+1):
        if ((df['close'][i]-df['open'][i]) / df['open'][i] > 0.09) and ((df['close'][i-1]-df['open'][i-1])/df['open'][i-1] < 0.02):
            signal_ls.append(1)
        else:
            signal_ls.append(0)
    assert(len(df) == len(signal_ls))
    df['1st_raise_limit'] = signal_ls

    return df[1:]

@metric
def add_trend_strength_factor(df, n, r=0):
    data = np.array(df['close'])
    fillna_ls = [np.nan]*(n-1)
    for i in range(data.shape[0]-n +1):
        window_data = data[i:i+n]
        path = 0
        for k in range(1, window_data.shape[0]):
            path = path + abs(window_data[k]-window_data[k-1])
        distance = window_data[-1]-window_data[0]
        trend_strength = round(distance/path, 4)
        fillna_ls.append(trend_strength)
    df['trend_strength_feature'] = fillna_ls

    return df


# The Chande Momentum Oscillator is a modified RSI. > 50 indicate overbought while < -50 indicate oversold
# ADX is used to determine the strength of a trend
def add_momentum_factor(df):
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['cmo'] = talib.CMO(df['close'], timeperiod=14)
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'],  timeperiod=14)
    return df


def add_cycle_indicator_factor(df):
    df['ht_trend'] = talib.HT_TRENDMODE(df['close'])
    df['ht_sine'], df['ht_leadsine'] = talib.HT_SINE(df['close'])
    df['ht_sine_feature'] = list(map(lambda x, y: int((x - y) > 0), df['ht_leadsine'], df['ht_sine']))
    return df


def extract_imfs(signal):
    eemd = EEMD()
    eemd.eemd(signal)
    res = eemd.get_imfs_and_residue()
    imfs = list(res)[0]
    # drop the noise/ high frequency imfs, may incur index error due to the signal is too simple
    # imfs_left = imfs[-5:]

    # return imfs_left # np.array type
    return imfs

@metric
def add_eemd_factor(df, window_len, col): # suppose window_len < len(df)
    data = np.array(df[col])  # numpy.ndarray
    data_windows = []
    for i in range(len(df) - window_len +1):
        data_windows.append(data[i:i+window_len])

    data_imfs = [extract_imfs(ele).T[-1] for ele in data_windows]
    min_len = np.min(np.array([len(ele) for ele in data_imfs]))
    data_imfs_min = np.array([ele[-min_len:] for ele in data_imfs])

    for k in range(min_len):
        df[col+'_imf'+str(k+1)] = [np.nan]*(window_len-1) + list(data_imfs_min[:, k])

    return df

# @metric
def add_predict_y(df, n, roc_min):  # 放宽对y的要求，比如，未来n日内第m日相对第1日涨幅超过5%就视为正例
    data = np.array(df['close'])
    y = []
    for i in range(data.shape[0]-n+1):
        window = data[i:i+n]
        roc_max = (np.max(window[1:]) - window[0])/window[0]
        if roc_max > roc_min:
            y.append(int(1))
        else:
            y.append(int(0))
    df['predict_ynm'] = y + [np.nan]*(n-1)
    return df
