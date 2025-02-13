#!/usr/bin/python
#-*- coding: utf-8 -*- 

import sys
from .sohu_quotation import SohuQuotation
from .sina_quotation import SinaQuotation
from .tencent_quotation import TencentQuotation
from .akshare_quotation import AkshareQuotation

class Quotation:
    """ 行情类 """

    def __init__(self):
        self.sohu = SohuQuotation()
        self.sina = SinaQuotation()
        self.tecent = TencentQuotation()
        self.akshare = AkshareQuotation()
        return

    def get_realtime_quotes(self, codes):
        """
        获取当前行情
        """
        d = self.sina.get_realtime_quotes(codes)
        return d

    def get_today_ticks(self, code):
        """
        获取当天tick数据
        """
        d = self.sohu.get_today_ticks(code)
        return d

    def get_daily_data(self, code, expire=60*6):
        """
        获取一支股票所有历史数据保存到本地
        """
        d = self.akshare.get_daily_data(code, expire)
        return d

    def get_tick_data(self, code, date, expire=60*24*365*10):
        """
        获取一支股票一天的tick数据保存到本地
        --------
        code: string,股票代码
        date: string,1900-01-01
        """
        d = self.akshare.get_tick_data(code, date, expire)
        return d

    def get_spot_data(self,  expire=60*6):
        """
        次返回所有沪深京 A 股上市公司的实时行情数据
        http://quote.eastmoney.com/center/gridlist.html#hs_a_board
        """
        d = self.akshare.get_spot_data(expire)
        return d

    #def get_one_realtime_quotes(self, code):
    #    """
    #    获取当前行情
    #    """
    #    d = self.eastmoney.get_realtime_quotes(code)
    #    return d

    #def get_today_shibor_ON(self):
    #    """
    #    获取今天的银行间拆借利率 隔夜(O/N)
    #    """
    #    d = self.tushare.get_today_shibor_ON()
    #    return d

    #def get_sina_dd(self, code, date='', vol=400):
    #    """
    #    大单交易数据
    #    """
    #    d = self.tushare.get_sina_dd(code, date, vol)
    #    return d

    #def get_hgt_capital(self):
    #    """
    #    得到当前沪股通资金情况
    #    """
    #    d = self.eastmoney.get_hgt_capital()
    #    return d

    #def get_hsgt_top(self, date_str, expire = 60*24*30):
    #    """
    #    得到沪股通、深股通 十大成交股
    #    """
    #    d = self.eastmoney.get_hsgt_top(date_str, expire)
    #    return d

    #def get_hsgt_his(self, days=30, market_type=1, expire = 60*6):
    #    """
    #    得到沪股通、深股通 历史资金数据
    #    资金单位百万
    #    """
    #    d = self.eastmoney.get_hsgt_his(days, market_type, expire)
    #    return d


    def get_trade_date(self, days = -1):
        """
        得到交易日期
        通过上证指数的日期来得到有交易的时间
        """
        d = self.get_h_data('000001')

        if days != -1:
            d = d.head(days)

        return list(d.index)

    def test(self):
        import tushare as ts
        #print(ts.get_hist_data('600848', ktype='1', start='2020-01-01'))
        print(ts.get_tick_data('sh',date='2017-12-12',src='tt'))


def main(argv):
    q = Quotation()

    #q.test()

    #r = q.get_realtime_quotes('sh')
    #for (k, v) in list(r.items()):
    #    string = v.__str__()
    #    print((string.encode('utf-8')))
    #d = q.get_realtime_quotes(['000001', '000002'])
    #print((len(d)))
    #for (k, v) in list(d.items()):
    #    print(k)
    #    string = v.__str__()
    #    print((string.encode('utf-8')))
    #    print((v.name.encode('utf-8')))


    #d = q.get_today_ticks('sh')
    #print((d.symbol))
    #print((d.df))

    d = q.get_daily_data('000001')
    d = q.get_daily_data('601328')
    d = q.get_daily_data('601288')
    d = q.get_daily_data('002254')
    print(d)
    #d = q.get_tick_data('000001', '2020-12-02')
    #print(d)
    #print((len(d)))

    ##d = q.get_one_realtime_quotes('131800')
    ##for (k,v) in list(d.items()):
    ##    string = v.__str__()
    ##    print((string.encode('utf-8')))

    ##d = q.get_stock_basics()
    ##print(d)
    ##print((d.index))
    ##print((len(d['name'])))

    #d = q.get_today_shibor_ON()
    #print(d)
    #d = q.get_hgt_capital()
    #print(d)

    #d = q.get_sina_dd('600340', date='2017-04-21', vol=400)
    #d.to_csv('tt.csv', sep='\t')
    #d = q.get_trade_date(30)
    #print(d)

    return

if __name__ == "__main__":
    main(sys.argv)
