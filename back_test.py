import pymongo
from pymongo import MongoClient
from core.multi_factors_dig import *
import glob
import os
import pandas as pd
import numpy as np
import json
import logging
import talib


def mongoClient(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection


def kelly_formula(p, b):  # p获胜率 b盈亏比（不含本金的赔率）
    return (p*b + p - 1)/b


def main():
    back_test_db = mongoClient('mongodb://localhost:27017/', 'Astock', 'market_daily_status')
    init_fund = 1000000
    left_fund = init_fund
    position_info = {}  # record the position in detail: which stocks and how much in hand
    fund_info = []  # record the fund varies in history
    select_stock_pool = set()

    # get the time line
    df = pd.read_csv('./data/000001.csv')
    df = df.rename(columns={'Unnamed: 0': 'datetime'})
    timeline = df['datetime'].tolist()
    for i in range(len(timeline) - 1):
        today = timeline[i]
        tomorrow = timeline[i + 1]
        stock_find = back_test_db.find({'datetime': today, 'score': {'$gt': 4}}).sort([('score', -1)])
        for item in stock_find:
            stock_name = item['name']
            stock_info = back_test_db.find({'datetime': tomorrow, 'name': stock_name})
            tomorrow_info = [ele for ele in stock_info][0]
            buy_price = tomorrow_info['open']
            if buy_price * 100 + 5 > left_fund:
                continue
            buy_num = min(np.floor(left_fund/buy_price/100), np.floor(10000/buy_price/100))
            if buy_num < 1:
                continue
            order_info = {'buy_price': buy_price,
                          'buy_num': buy_num*100,
                          'buy_reason': 'strategy_name',
                          'buy_time': tomorrow_info['datetime']
                          }
            















if __name__ == '__main__':
    main()