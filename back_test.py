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


def calculate_win_probability(history_order, position_info):
    pass


def update_trade_info(position_info, position_fund_info, back_test_db, tomorrow):
    stock_hold = [code for code, info in position_info.items()]
    price_info = back_test_db.find({'datetime': tomorrow, 'name': {'$in': stock_hold}})
    value = 0
    for item in price_info:
        value = value + item['close'] * position_fund_info[item['name']]['num']
    position_fund_info[tomorrow] = value

    return 0


def main():
    back_test_db = mongoClient('mongodb://localhost:27017/', 'Astock', 'market_daily_status')

    # Initialize the trade params
    init_fund = 1000000
    left_fund = init_fund
    position_info = {}  # record the position in detail: which stocks and how much in hand
    position_fund_info = {}  # record the fund in position varies in history
    monetary_fund = {}  # record the static fund in history
    history_order = []

    # get the time line
    df = pd.read_csv('./data/000001.csv')
    df = df.rename(columns={'Unnamed: 0': 'datetime'})
    timeline = df['datetime'].tolist()[-2600:]  # roughly last 10 yrs
    position_fund_info[timeline[0]] = 0
    monetary_fund[timeline[0]] = init_fund
    print('backtest start time:', timeline[0])

    # 一般按照盈亏比2进行止盈止损 以下策略判断为底部信号 因此绝对止损，不止盈
    # 每天检查市场信号和仓位信息，进行先卖后买操作
    for i in range(len(timeline) - 1):
        today = timeline[i]
        tomorrow = timeline[i + 1]

        # 检查仓位信息，确定需要止损的股票，并且操作止损
        for code, info in position_info.items():
            close_price = 0 # mongo return value
            if (close_price - position_info[code]['price']) / position_info[code]['price'] < -0.2:  # %20 stop loss
                open_price = 0 # mongo return value

                # status update
                monetary_fund[tomorrow] = monetary_fund[today] + open_price * position_info[code]['num']
                position_info.pop(code, None)
                # del position_info[code]
                order_info = {
                                'price': position_info[code]['price'],
                                'num': position_info[code]['num'],
                                'reason': 'stop loss',
                                'status': 'success',
                                'operation': 'sell',
                                'datetime': tomorrow,
                                'code': code
                                }
                history_order.append(order_info)

        # 检查市场信号，确定要进场的股票，并且操作买入（手续费每笔5）
        stock_find = back_test_db.find({'datetime': today, 'score': {'$gt': 4}}).sort([('score', -1)])
        if stock_find.count() == 0:  # no market signal today
            update_trade_info(position_info, position_fund_info, back_test_db, tomorrow)
            monetary_fund[tomorrow] = monetary_fund[today]
            continue

        for item in stock_find:
            stock_name = item['name']
            stock_info = back_test_db.find({'datetime': tomorrow, 'name': stock_name})
            tomorrow_info = [ele for ele in stock_info][0]
            buy_price = tomorrow_info['open']
            if buy_price * 100 + 5 > left_fund:
                order_info = {
                    'price': None,
                    'num': None,
                    'reason': 'insufficient fund',
                    'status': 'fail',
                    'operation': 'buy',
                    'datetime': tomorrow,
                    'code': stock_name
                }
                history_order.append(order_info)
                continue

            buy_num = min(np.floor(left_fund/buy_price/100), np.floor(10000/buy_price/100))  # 等权买入10000
            order_info = {
                            'price': buy_price,
                            'num': buy_num * 100,
                            'reason': 'strategy_name',
                            'status': 'success',
                            'operation': 'buy',
                            'datetime': tomorrow,
                            'code': stock_name
                          }
            history_order.append(order_info)
            left_fund = left_fund - buy_price * buy_num * 100
            if stock_name in position_info:
                total_num = buy_num*100 + position_info[stock_name]['num']
                total_cost = (position_info[stock_name]['num']*position_info[stock_name]['price']+buy_price*buy_num*100)
                avg_price = round(total_cost / total_num, 2)
                position_info[stock_name]['price'] = avg_price
                position_info[stock_name]['num'] = total_num
            else:
                position_info[stock_name] = {'price': buy_price, 'num': buy_num*100}

        update_trade_info(position_info, position_fund_info, back_test_db, tomorrow)
        monetary_fund[tomorrow] = left_fund


if __name__ == '__main__':
    main()