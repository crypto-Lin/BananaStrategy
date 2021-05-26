# load and transform data for training

from core.multi_factors_dig import *

import glob
import os
import pandas as pd
import json
import logging

pd.options.mode.chained_assignment = None  # default='warn'

data_path = './data/'
# data_path = './dataset/A_stock_5d_original'
configs = json.load(open('config.json', 'r'))


def main():

    logging.basicConfig(filename='data_transform.log', filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    feature_select = configs['factor_feature_extract_all']
    predict_yn = configs['predict_timeperiod']
    counter = 0
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    train_test_split = configs['model_params']['train_test_split']

    for csvfile in glob.glob(os.path.join(data_path, '*.csv')):
        tmpdf = pd.read_csv(csvfile)
        if len(tmpdf) < 200:
            continue
        try:
            newdf = add_macd_factor(tmpdf)
            newdf = add_kd_factor(newdf)
            newdf = add_rsi_factor(newdf)
            newdf = add_ma_factor(newdf, 10, 20)
            newdf = add_ma_factor(newdf, 5, 10)
            newdf = add_ema_diff_factor(newdf, 10, 1)
            newdf = add_ema_diff_factor(newdf, 20, 1)
#            newdf = add_atr_factor(newdf)
            newdf = add_bbands_factor(newdf)

            newdf = add_up_pattern_recognition_factor(newdf)
            newdf = add_down_pattern_recognition_factor((newdf))
            newdf = add_cycle_indicator_factor(newdf)
            newdf = add_eemd_factor(newdf, 100, ['close'])
#            newdf = add_macd_cross_factor(newdf)
#            newdf = add_ma_cross_factor(newdf, 5, 10)
#            newdf = add_ma_cross_factor(newdf, 10, 20)
#            newdf = add_ma_cross_factor(newdf, 50, 100)
#            newdf = add_ma_cross_factor(newdf, 50, 200)

            for k in predict_yn:
                newdf = add_roc_factor(newdf, k)
           
            newdf = newdf.dropna(axis=0, how='any')

            newdf = add_time_factor(newdf)
            newdf = add_first_raising_limit_factor(newdf)

            data_for_train = newdf[feature_select]
            test_size = int(len(newdf)*train_test_split)
            df_train = pd.concat([df_train, data_for_train[:-test_size]])
            df_test = pd.concat([df_test, data_for_train[-test_size:]])
            counter = counter + len(data_for_train)

        except Exception as e:
            logging.error(e)
            logging.info(csvfile.split('/')[-1])
        else:
            logging.info(csvfile.split('/')[-1])

        print(counter)
        #break

    if not os.path.exists('./dataset/A_stock_5d'):
        os.makedirs('./dataset/A_stock_5d')

    # represent the macd cross signal occurs
    # df_test = df_test[(df_test['macd_cross_up_signal'] == 1) | (df_test['macd_cross_down_signal'] == 1)]
    df_test = df_test.reset_index(drop=True)

    # df_train = df_train[(df_train['macd_cross_up_signal'] == 1) | (df_train['macd_cross_down_signal'] == 1)]
    df_train = df_train.reset_index(drop=True)

    df_train.to_csv('./dataset/A_stock_daily/train.csv')
    df_test.to_csv('./dataset/A_stock_daily/test.csv')
    # df_train.to_csv('./dataset/A_stock_5d/train.csv')
    # df_test.to_csv('./dataset/A_stock_5d/test.csv')

    print('total number of trainset: {}'.format(counter))


if __name__ == '__main__':
    main()
