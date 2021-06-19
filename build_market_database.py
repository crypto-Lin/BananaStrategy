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

#data_path = 'data'
data_path = './dataset/A_stock_daily/test_with_xgb_predict.csv'


def mongoClient(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection


def main():
    logging.basicConfig(filename='data_import_mongo.log', filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    back_test_db = mongoClient('mongodb://localhost:27017/', 'Astock', 'xgb_daily_status')
    df = pd.read_csv(data_path)
    count = 0
    try:
        for index, row in df.iterrows():
            back_test_db.insert_one(row.to_dict())
            count = count + 1
            if(count%1000==0):
                print(count)
        #back_test_db.insert_many([row.to_dict() for index, row in df.iterrows()])
    except Exception as e:
        logging.error('insert mongodb error!')
        logging.info(e)

#     for csvfile in glob.glob(os.path.join(data_path, '*.csv')):
#         logging.info(csvfile)
#
#         df = pd.read_csv(csvfile)
#         df = df.rename(columns={'Unnamed: 0': 'datetime'})
#         name = csvfile.split('/')[-1].split('.')[0]
#         df['name'] = [name] * len(df)
#         print(name)
#         try:
#             df = try_bottom_strategy(df)
# #            print(df.head())
#         except Exception as e:
#             logging.error(e)
#             logging.info(csvfile)
#         try:
#             back_test_db.insert_many([row.to_dict() for index, row in df.iterrows()])
#         except Exception as e:
#             logging.error('insert mongodb error!')
#             logging.info(e)
#             logging.info(csvfile)


if __name__ == '__main__':
    main()

