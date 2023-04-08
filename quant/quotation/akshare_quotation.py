#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import os
import traceback

import akshare as ak
from akshare import stock_zh_a_daily
import pandas as pd
import numpy as np
import requests
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from utils.symbol import symbol_of, is_index
import utils.const as CT
import utils.date_time as date_time
from utils import utils as UT

class AkshareQuotation():
    """历史数据"""

    def __init__(self):
        return

    def get_daily_data(self, code, expire=60*6):
        """
        获取一支股票所有历史数据保存到本地
        目标地址: https://finance.sina.com.cn/realstock/company/sh600006/nc.shtml(示例)
        """
        UT.check_dir(CT.DAILY_DIR)
        symbol = self._code_to_symbol(code)
        file_path = "%s%s%s" % (CT.DAILY_DIR, symbol, CT.FILE_SUFFIX)
        expired = UT.check_file_expired(file_path, expire)

        if expired or not os.path.exists(file_path):
            start_date = CT.START
            end_date = date_time.get_today_str()
            # adjust = 'qfq'
            adjust = 'hfq'
            d = None
            if is_index(code):
                d = ak.stock_zh_index_daily(symbol)
            else:
                print(symbol, start_date, end_date, adjust)
                try:
                    d = ak.stock_zh_a_daily(symbol, start_date, end_date, adjust)
                except Exception as e:
                    print("get_daily_data exception, code=%s, e=%s, %s" % (code, repr(e), traceback.format_exc()))
                    #print(traceback.format_exc())
                #d = ak.stock_zh_a_hist(code, 'daily', start_date, end_date, adjust)
            if d is None:
                return d
            # d.to_csv(file_path, sep='\t')
            d.to_csv(file_path, sep=CT.COLUMN_SEP)

        if not os.path.exists(file_path):
            return None
        #d = pd.read_csv(file_path, sep='\t', index_col=0)
        #d = pd.read_csv(file_path, sep='\t', skiprows=0, parse_dates=True, header=0, index_col=0)
        d = pd.read_csv(file_path, sep=CT.COLUMN_SEP, skiprows=0, parse_dates=True, header=0, index_col=0)
        return d

    def get_spot_data(self,  expire=60*6):
        """
        次返回所有沪深京 A 股上市公司的实时行情数据
        http://quote.eastmoney.com/center/gridlist.html#hs_a_board
        """
        UT.check_dir(CT.BASICS_DIR)
        file_path = "%s%s%s" % (CT.BASICS_DIR, CT.SPOT_FILE, CT.FILE_SUFFIX)
        expired = UT.check_file_expired(file_path, expire)

        if expired or not os.path.exists(file_path):
            d = ak.stock_zh_a_spot_em()
            if d is None:
                return d
            # d.to_csv(file_path, sep='\t')
            d.to_csv(file_path, sep=CT.COLUMN_SEP)

        if not os.path.exists(file_path):
            return None
        # d = pd.read_csv(file_path, sep='\t', index_col=0)
        # d = pd.read_csv(file_path, sep='\t', skiprows=0, parse_dates=True, header=0, index_col=0)
        d = pd.read_csv(file_path, dtype={"代码": np.string_}, sep=CT.COLUMN_SEP, skiprows=0, parse_dates=True, header=0, index_col=0)
        return d

    #def get_daily_data(self, code, start_date='', end_date='', adjust='qfq'):
    #    """
    #    获取一支股票天级历史数据保存到本
    #    """
    #    symbol = self._code_to_symbol(code)
    #    if start_date == '':
    #        start_date = CT.START
    #    if end_date == '':
    #        end_date = date_time.get_today_str()
    #    stock_zh_a_daily_hfq_df = ak.stock_zh_a_daily(symbol, start_date, end_date, adjust)
    #    return stock_zh_a_daily_hfq_df

    def get_tick_data(self, code, trade_date, expire=60*24*365*10):
        """
        获取一支股票一天的tick数据保存到本地
        """
        UT.check_dir(CT.TICK_DIR + code)
        file_path = CT.TICK_DIR + code + '/' + trade_date
        symbol = self._code_to_symbol(code)
        trade_date = date_time.date_to_str(date_time.str_to_date(trade_date), '%Y%m%d')
        expired = UT.check_file_expired(file_path, expire)
        if expired or not os.path.exists(file_path):
            d = ak.stock_zh_a_tick_tx(symbol, trade_date)
            #过掉当天没数据的
            if d is None or len(d) < 10:
                return None
            d.to_csv(file_path, sep='\t')

        if not os.path.exists(file_path):
            return None

        #d = pd.read_csv(file_path, sep='\t', index_col=1)
        d = pd.read_csv(file_path, sep='\t', skiprows=0, parse_dates=True, header=0, index_col=0)

        #过掉当天没数据的
        if d is None or len(d) < 10:
            return None
        return d

    def get_minute_data(self, code, period='1', adjust="", expire=60*6):
        """
        获取一支股票分钟级数据保存到本地
        """
        UT.check_dir(CT.MINUTE_DIR + '/' + period)
        file_path = CT.MINUTE_DIR + '/' + period + '/' + code

        expired = UT.check_file_expired(file_path, expire)
        if expired or not os.path.exists(file_path):
            symbol = self._code_to_symbol(code)
            start_date = CT.START
            end_date = date_time.get_today_str()
            adjust = 'qfq'
            # d = ak.stock_zh_a_minute(symbol, period, adjust)
            d = self.stock_zh_a_minute(symbol, period, adjust)
            if d is None:
                return d
            d.to_csv(file_path, sep='\t')

        if not os.path.exists(file_path):
            return None
        #d = pd.read_csv(file_path, sep='\t', index_col=1)
        d = pd.read_csv(file_path, sep='\t', skiprows=0, parse_dates=True, header=0, index_col=0)
        return d

    def stock_zh_a_minute(
            self, symbol: str = "sh600751", period: str = "5", adjust: str = ""
    ) -> pd.DataFrame:
        """
        修改 akshare stock_zh_a_minute 参数，调大 datalen 参数,增加 ma 参数
        max 37676
        """
        url = (
            "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/CN_MarketDataService.getKLineData"
        )
        params = {
            "symbol": symbol,
            "scale": period,
            # "datalen": "20000",
            "datalen": "34000",
            "ma": "no",
        }
        r = requests.get(url, params=params)
        temp_df = pd.DataFrame(json.loads(r.text.split("=(")[1].split(");")[0])).iloc[:, :6]
        try:
            stock_zh_a_daily(symbol=symbol, adjust="qfq")
        except:
            return temp_df
        if adjust == "":
            return temp_df

        if adjust == "qfq":
            temp_df[["date", "time"]] = temp_df["day"].str.split(" ", expand=True)
            need_df = temp_df[temp_df["time"] == "15:00:00"]
            need_df.index = need_df["date"]
            stock_zh_a_daily_qfq_df = stock_zh_a_daily(symbol=symbol, adjust="qfq")
            result_df = stock_zh_a_daily_qfq_df.iloc[-len(need_df):, :]["close"].astype(
                float
            ) / need_df["close"].astype(float)
            temp_df.index = pd.to_datetime(temp_df["date"])
            merged_df = pd.merge(temp_df, result_df, left_index=True, right_index=True)
            merged_df["open"] = merged_df["open"].astype(float) * merged_df["close_y"]
            merged_df["high"] = merged_df["high"].astype(float) * merged_df["close_y"]
            merged_df["low"] = merged_df["low"].astype(float) * merged_df["close_y"]
            merged_df["close"] = merged_df["close_x"].astype(float) * merged_df["close_y"]
            temp_df = merged_df[["day", "open", "high", "low", "close", "volume"]]
            temp_df.reset_index(drop=True, inplace=True)
            return temp_df
        if adjust == "hfq":
            temp_df[["date", "time"]] = temp_df["day"].str.split(" ", expand=True)
            need_df = temp_df[temp_df["time"] == "15:00:00"]
            need_df.index = need_df["date"]
            stock_zh_a_daily_qfq_df = stock_zh_a_daily(symbol=symbol, adjust="hfq")
            result_df = stock_zh_a_daily_qfq_df.iloc[-len(need_df):, :]["close"].astype(
                float
            ) / need_df["close"].astype(float)
            temp_df.index = pd.to_datetime(temp_df["date"])
            merged_df = pd.merge(temp_df, result_df, left_index=True, right_index=True)
            merged_df["open"] = merged_df["open"].astype(float) * merged_df["close_y"]
            merged_df["high"] = merged_df["high"].astype(float) * merged_df["close_y"]
            merged_df["low"] = merged_df["low"].astype(float) * merged_df["close_y"]
            merged_df["close"] = merged_df["close_x"].astype(float) * merged_df["close_y"]
            temp_df = merged_df[["day", "open", "high", "low", "close", "volume"]]
            temp_df.reset_index(drop=True, inplace=True)
            return temp_df


    def _code_to_symbol(self, code):
        return symbol_of(code)

def main(argv):
    q = AkshareQuotation()

    #r = q.get_daily_data('000001')
    #print(r)
    r = q.get_daily_data('sh')
    print(r)

    #r = q.get_spot_data()
    #print(r)

    #r = q.get_minute_data('000001')
    #print(r)
    #r = q.get_minute_data('600519')
    #print(r)
    #r = q.get_minute_data('sh')
    #print(r)

    #r = q.get_tick_data('000001', '2023-02-09')
    #print(r)

if __name__ == '__main__':
    main(sys.argv)
