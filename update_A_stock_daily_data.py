# update the total securities daily data

import os.path
from jqdatasdk import *
from datetime import *
import datetime
import pandas as pd

import logging

data_path = '/Users/li/A-stock-life-jacket/data/'

def main():

    logging.basicConfig(filename='update_A_stock_daily_data.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    auth('15182289113', 'Since1991')
    assert (get_query_count()['total'] > 100000)

    today = str(datetime.datetime.now()).split(' ')[0]
    stock_table = get_all_securities(types=['stock'])
    stock_table.to_csv('/Users/li/A-stock-life-jacket/data/stock_table_original.csv')


    for i in range(len(stock_table)):

        logging.info(i)
        logging.info('{} is in process.'.format(stock_table['display_name'][i]))
        #logging.warning('logging starts here!')
        start_date = stock_table['start_date'][i]

        if not os.path.isfile(data_path + stock_table.index[i].split('.')[0] + '.csv'):
            # file now existed!
            df = get_price(security=stock_table.index[i],
                           start_date=start_date,
                           end_date=today,
                           frequency='daily',
                           fields=None,
                           skip_paused=True,
                           fq='pre', count=None)
            df.to_csv(data_path + stock_table.index[i].split('.')[0] + '.csv')
            continue

        tmp_df = pd.read_csv(data_path + stock_table.index[i].split('.')[0] + '.csv')

        try:
            last_date = tmp_df.iloc[-1].values[0]
            last_date = pd.to_datetime(last_date)
            # last_date = last_date + timedelta(days=1)
        except:
            last_date = start_date

        df = get_price(security=stock_table.index[i],
                       start_date=last_date,
                       end_date=today,
                       frequency='daily',
                       fields=None,
                       skip_paused=True,
                       fq='pre', count=None)

        tmp_df = tmp_df.set_index('Unnamed: 0')
        bigdata = tmp_df[:-1].append(df, ignore_index=False)
        bigdata = bigdata.drop_duplicates()
        bigdata.index = pd.to_datetime(bigdata.index)
        bigdata = bigdata.sort_index()

        bigdata.to_csv(data_path + stock_table.index[i].split('.')[0] + '.csv')
        #break


if __name__ == '__main__':
    main()