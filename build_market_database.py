# For the sake of backtest, it is necessary to frozen the market status at different time slice.
# This module is to import stock info to mongodb.

import pymongo
from pymongo import MongoClient
from core.multi_factors_dig import *
import glob
import os
import pandas as pd
import json
import logging
import talib

data_path = 'data'


def mongoClient(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection


def main():
    logging.basicConfig(filename='data_transform.log', filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    back_test_db = mongoClient('mongodb://localhost:27017/', 'Astock', 'market_daily_status')

    for csvfile in glob.glob(os.path.join(data_path, '*.csv')):
        logging.info(csvfile)

        df = pd.read_csv(csvfile)
        df = df.rename(columns={'Unnamed: 0': 'datetime'})
        try:
            df = try_bottom_strategy(df)
        except Exception as e:
            logging.error(e)
            logging.info(csvfile)
        try:
            back_test_db.insert_many([row.to_dict() for index, row in df.iterrows()])
        except Exception as e:
            logging.error('insert mongodb error!')
            logging.info(e)
            logging.info(csvfile)


if __name__ == '__main__':
    main()

