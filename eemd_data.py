# load and transform data for training

from core.multi_factors_dig import *

import glob
import os
import pandas as pd
import json
import logging

pd.options.mode.chained_assignment = None  # default='warn'

data_path = 'data'
configs = json.load(open('config.json', 'r'))


def main():
    logging.basicConfig(filename='data_transform.log', filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    predict_yn = configs['predict_timeperiod']
    counter = 0
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    train_test_split = configs['model_params']['train_test_split']

    if not os.path.exists('./dataset/emd_data'):
        os.makedirs('./dataset/emd_data')

    for csvfile in glob.glob(os.path.join(data_path, '*.csv')):
        tmpdf = pd.read_csv(csvfile)
        if len(tmpdf) < 1000:
            continue
        try:
            newdf = add_eemd_factor(tmpdf, 100, 'close')
            # newdf = add_eemd_factor(newdf, 100, 'volume')
            # print(newdf.head())
            newdf = add_trend_strength_factor(newdf, 100)

            newdf = add_predict_y(newdf, 5, 0.03)
            # print(newdf.head())
            for k in predict_yn:
                newdf = add_roc_factor(newdf, k)

            newdf = newdf.dropna(axis=0, how='any')

            data_for_train = newdf.reset_index(drop=True).to_csv('./dataset/emd_data/' + csvfile.split('/')[-1])

            counter = counter + len(data_for_train)

        except Exception as e:
            logging.error(e)
            logging.info(csvfile.split('/')[-1])
        else:
            logging.info(csvfile.split('/')[-1])

        print(counter)
        break

    print('total number of trainset: {}'.format(counter))


if __name__ == '__main__':
    main()