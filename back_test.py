import pymongo
from pymongo import MongoClient
from core.multi_factors_dig import *
import glob
import os
import pandas as pd
import json
import logging
import talib



def mongoClient(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection


def main():
    # get the time line
    df = pd.read_csv('./data/00001.csv')

    back_test_db = mongoClient('mongodb://localhost:27017/', 'Astock', 'market_daily_status')
    init_fund = 1000000
    select_stock_pool = set()









if __name__ == '__main__':
    main()