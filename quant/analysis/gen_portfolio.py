#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import os

import qlib
from qlib.constant import REG_CN
from qlib.data import D

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from quotation.quotation import Quotation
from utils.symbol import upper_symbol_of
import utils.const as CT
import utils.date_time as date_time
from utils import utils as UT



def gen_analysis_portfolio():
    """
    生成 A 股分析的投资组合
    """
    UT.check_dir(CT.MY_QLIB_INSTRUMENTS_DIR)
    q = Quotation()
    spot_data = q.get_spot_data()


    """ 壳资源，规避了中国股市的壳价值污染问题，在实证中剔除了市值最低的30%的股票。"""
    spot_data.dropna(subset=["总市值"], inplace=True)
    print(spot_data.head(50))
    market_value = spot_data["总市值"]
    #market_value.dropna(inplace=True)
    #market_value = market_value.sort_values().reindex(range(market_value.count()))
    market_value = market_value.sort_values().reset_index(drop=True)
    print(int(market_value.count() * 0.3))
    min_market_value = market_value[int(market_value.count() * 0.3)]
    print("min_market_value:", min_market_value)
    spot_data = spot_data[spot_data["总市值"] > min_market_value]
    portfolio_data_list = []
    print(spot_data["代码"])

    """ 过滤无数据 """
    portfolio_symbol_list = []
    all_instruments = get_all_instruments()
    for symbol in spot_data["代码"].apply(upper_symbol_of).to_list():
        if symbol in all_instruments:
            portfolio_symbol_list.append(symbol)

    for symbol in portfolio_symbol_list:
        portfolio_data_list.append("%s\t%s\t%s" % (symbol, CT.START, date_time.get_today_str()))
    file_path = CT.MY_QLIB_INSTRUMENTS_DIR + "analysis.txt"
    with open(file_path, "w") as f:
        f.write("\n".join(portfolio_data_list))

def get_all_instruments():
    provider_uri = '/Users/zhangyunsheng/Dev/sonata/data/myqlib'  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    instruments = D.instruments(market='all')
    return D.list_instruments(instruments=instruments, as_list=True)

if __name__ == '__main__':
    gen_analysis_portfolio()
