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
import csv
from core.utils import metric
pd.options.mode.chained_assignment = None  # default='warn'


def mongoClient(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection


def kelly_formula(p, b):  # p获胜率 b盈亏比（不含本金的赔率）
    return (p*b + p - 1)/b


def calculate_win_probability(history_order):
    sdf = history_order[history_order['status'] == 'success']
    order_count = len(sdf[sdf['operation'] == 'buy'])
    lose_count = 0
    win_count = 0
    
    for k, g in sdf.groupby('code'):
        g['datetime'] = pd.to_datetime(g['datetime'])
        g = g.sort_values('datetime')
        
        count = 0
        flag = True
        for i in range(len(g)):
            if g['operation'].values[i] == 'buy':
                count = count + 1
                continue
            if g['operation'].values[i] == 'sell':
                flag = False
                if g['reason'].values[i] == 'stop_loss':
                    lose_count = lose_count + count
                else:
                    win_count = win_count + count
                count = 0
        if flag:
            win_count = win_count + count
            flag = True
    try:
        assert (order_count == (win_count + lose_count))
    except Exception as e:
        print(order_count, win_count, lose_count)
        raise e
        
    win_rate = round(win_count / order_count, 2)
    return order_count, win_rate


def evaluate_fund_curve(position_fund_info, monetary_fund):
    try:
        assert(len(position_fund_info.items()) == len(monetary_fund.items()))
    except Exception as e:
        print('持仓资金信息和空仓资金信息时间不对齐')
        raise e
    df = pd.DataFrame([(k, v, monetary_fund[k]) for k, v in position_fund_info.items()],
                      columns=['datetime', 'position_fund', 'short_position_fund'])
    df['fund'] = df['position_fund'] + df['short_position_fund']
    df = df.round(2)
    dividend = df['fund'].values
    max_drawback = -100 * np.min([((dividend[i+1:]-dividend[i])/dividend[i]).min() for i in range(len(df)-1)])

    # sharpe ratio annualized
    rf = 0  # free risk rate
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['dr'] = talib.ROCP(df['fund'], timeperiod=1)
    df = df.dropna(how='any')
    df_sharpe = df[['datetime', 'dr']].set_index('datetime').groupby(pd.Grouper(freq='Y')).apply(
        lambda x: (x.mean()-rf) / x.std() * np.sqrt(len(x)))
    df_sharpe = df_sharpe.rename(columns={'dr': 'annualized_sharpe_ratio'})

    # annualized return 累计收益率
    total_r = round(df.iloc[-1]['fund']/df.iloc[0]['fund'], 2)
    yrs = round(int(str(df.iloc[-1]['datetime'] - df.iloc[0]['datetime']).split(' ')[0])/365, 2)
    annualized_r = pow(total_r, round(1/yrs, 2)) - 1

    df.to_csv('./dataset/position_info.csv')
    return round(annualized_r*100, 2), round(max_drawback, 2), df_sharpe.round(2)


def update_trade_info(position_info, position_fund_info, back_test_db, today, tomorrow):
    stock_hold = [code for code, info in position_info.items()]
    try:
        price_info = back_test_db.find({'datetime': tomorrow, 'name': {'$in': stock_hold}})
    except Exception as e:
        position_fund_info[tomorrow] = position_fund_info[today]
        return 0

    # update position_info firstly
    for item in price_info:
        position_info[item['name']]['market_p'] = item['close']

    # then update position fund info
    value = 0
    for code, v in position_info.items():
        value = value + position_info[code]['market_p'] * position_info[code]['num']
    position_fund_info[tomorrow] = value

    return 0

@metric
def main():
    logging.basicConfig(filename='back_test.log', filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    # back_test_db = mongoClient('mongodb://localhost:27017/', 'Astock', 'market_daily_status')
    back_test_db = mongoClient('mongodb://localhost:27017/', 'Astock', 'xgb_daily_status')

    # Initialize the trade params
    init_fund = 10000000
    left_fund = init_fund
    position_info = {}  # record the position in detail: which stocks and how much in hand
    position_fund_info = {}  # record the fund in position varies in history
    monetary_fund = {}  # record the static fund in history
    history_order = []

    # get the time line
    df1 = pd.read_csv('./data/000001.csv').set_index('Unnamed: 0')
    df2 = pd.read_csv('./data/000002.csv').set_index('Unnamed: 0')
    df = pd.concat([df1, df2], axis=1)
    # timeline = df.index.values[-1000:]
    timeline = df['2018-06-01':].index.values
    position_fund_info[timeline[0]] = 0
    monetary_fund[timeline[0]] = init_fund
    print('backtest start time:', timeline[0])

    # 一般按照盈亏比2进行止盈止损 以下策略判断为底部信号 因此绝对止损，不止盈
    # 每天检查市场信号和仓位信息，进行先卖后买操作
    for i in range(len(timeline) - 1):
        today = timeline[i]
        tomorrow = timeline[i + 1]
        print(today)
        # 检查仓位信息，确定需要止损的股票，并且操作止损
        for code in list(position_info.keys()):

            # 止损操作
            if (position_info[code]['market_p'] - position_info[code]['price']) / position_info[code]['price'] < -0.05:  # %20 stop loss
                try:
                    open_price = [item for item in back_test_db.find({'datetime': tomorrow, 'name': code})][0]['open'] # mongo return value
                except Exception as e:
                    logging.info('mongo not found.')
                    logging.info(tomorrow + ' ' + code)
                    continue

                # status update
                left_fund = left_fund + open_price * position_info[code]['num']
                order_info = {
                                'price': position_info[code]['price'],
                                'num': position_info[code]['num'],
                                'reason': 'stop_loss',
                                'status': 'success',
                                'operation': 'sell',
                                'datetime': tomorrow,
                                'code': code
                                }
                history_order.append(order_info)
                position_info.pop(code, None)
                # del position_info[code]
                continue

            # 止盈操作
            if (position_info[code]['market_p'] - position_info[code]['price']) / position_info[code]['price'] > 0.1 :
                try:
                    open_price = [item for item in back_test_db.find({'datetime': tomorrow, 'name': code})][0]['open']  # mongo return value
                except Exception as e:
                    logging.info('mongo not found.')
                    logging.info(tomorrow + ' ' + code)
                    continue
            
            #     # status update
                left_fund = left_fund + open_price * position_info[code]['num']
                order_info = {
                     'price': position_info[code]['price'],
                     'num': position_info[code]['num'],
                     'reason': 'stop_profit',
                     'status': 'success',
                     'operation': 'sell',
                     'datetime': tomorrow,
                     'code': code
                 }
                history_order.append(order_info)
                position_info.pop(code, None)

        # 检查市场信号，确定要进场的股票，并且操作买入（手续费每笔5）
        # stock_find = back_test_db.find({'datetime': today, 'score': {'$gt': 4}}).sort([('score', -1)])
        stock_find = back_test_db.find({'datetime': today, 'xgb_predict': {'$gt': 0}}).sort([('xgb_predict_proba', -1)]).limit(100)
        # if back_test_db.count_documents({'datetime': today, 'score': {'$gt': 4}}) == 0:  # no market signal today
        if back_test_db.count_documents({'datetime': today, 'xgb_predict': {'$gt': 0}}) == 0:  # no market signal today
            update_trade_info(position_info, position_fund_info, back_test_db, today, tomorrow)
            monetary_fund[tomorrow] = left_fund
            continue

        for item in stock_find:
            stock_name = item['name']
            stock_info = back_test_db.find({'datetime': tomorrow, 'name': stock_name})
#            print(tomorrow, stock_name)
            try:
                tomorrow_info = [ele for ele in stock_info][0]
            except Exception as e:
                logging.info('mongo not found.')
                logging.info(tomorrow+' '+stock_name)
                continue
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
                position_info[stock_name] = {'price': buy_price, 'num': buy_num*100, 'market_p': buy_price}
        # print(position_info)
        update_trade_info(position_info, position_fund_info, back_test_db, today, tomorrow)
        monetary_fund[tomorrow] = left_fund

    # evaluate the strategy
    # convert the history_order to dataframe
    keys = history_order[0].keys()
    with open('./dataset/history_order.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(history_order)
    df_history_order = pd.read_csv('./dataset/history_order.csv')

    print("backtest evaluation: ")
    try:
        tot_order, win_rate = calculate_win_probability(df_history_order)
        print("total order num: ", tot_order)
        print("win_rate: ", win_rate)
    except:
        pass
    annualized_r, max_drawback, df_sharpe = evaluate_fund_curve(position_fund_info, monetary_fund)
    print("annualized return: {} %".format(annualized_r))
    print("max drawback: ", max_drawback)
    print("annualized sharpe: ", df_sharpe)


if __name__ == '__main__':
    main()
