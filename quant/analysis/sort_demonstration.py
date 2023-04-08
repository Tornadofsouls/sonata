#!/usr/bin/python
import json
import pickle
import sys
import os
import copy
from collections import OrderedDict
from typing import Union, Text
import pandas as pd
import qlib
import qlib.contrib.report as qcr
from qlib.backtest import executor, backtest
from qlib.backtest.decision import TradeDecisionWO, OrderDir, Order
from qlib.backtest.position import Position
from qlib.constant import REG_CN
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report import analysis_position
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.data.dataset import DataHandlerLP, DatasetH
from qlib.model.base import BaseModel
from qlib.utils.time import Freq

import multiprocessing

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import utils.const as CT
from utils import utils as UT
from utils.logger import Logger
import utils.date_time as DT

GROUP_NUM = 10
GROUP_NO = 0
PROCESS_NUM = 4

FACTORS_NAME = ['turnover']

INSTRUMENTS = 'analysis'
#INSTRUMENTS = 'test'
BENCHMARK = ['SH000001']
SEGMENTS = {
    "train": ("2015-01-01", "2015-03-31"),
    "valid": ("2019-01-01", "2019-03-31"),
    "test": ("2022-01-01", "2022-12-31")
}
REPORT_PATH = CT.ANALYSIS_DIR + 'turnover/'

class SortFeature(DataHandlerLP):
    def __init__(self,
                 instruments='sh000300',
                 start_time=None,
                 end_time=None,
                 freq='day',
                 infer_processors=[],
                 learn_processors=[],
                 fit_start_time=None,
                 fit_end_time=None,
                 process_type=DataHandlerLP.PTYPE_A,
                 filter_pipe=None,
                 inst_processor=None,
                 **kwargs,
                 ):
        data_loader = {
            'class': 'QlibDataLoader',
            'kwargs': {
                'config': {
                    'feature': self.get_feature_config(),
                    'label': kwargs.get('label', self.get_label_config()),  # label可以自定义，也可以使用初始化时候的设置
                },
                'filter_pipe': filter_pipe,
                'freq': freq,
                'inst_processor': inst_processor,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def get_feature_config(self):
        #turnover_mean = 'Mean($turnover, 21)/Mean($turnover, 252)'
        turnover_mean = 'Mean($turnover, 21)'

        return [turnover_mean], FACTORS_NAME

    def get_label_config(self):
        return ['(Ref($close, -30) - $close) / $close'], ['LABEL']

class SortModel(BaseModel):

    def __init__(self, group_no):
        self.sort_factor_name1 = FACTORS_NAME[0]
        self.groups_num = GROUP_NUM
        #self.group_no = GROUP_NO
        self.group_no = group_no
        super(SortModel, self).__init__()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        #dl_test = dataset.prepare(segment, col_set=["CHANGE", "CHANGE_MEAN"], data_key=DataHandlerLP.DK_I)
        dl_test = dataset.prepare(segment, col_set="__all", data_key=DataHandlerLP.DK_I)
        #print("dl_test", dl_test)
        pred_data = []

        one_day_data = {}
        last_day = None
        for row_index, row in dl_test.iterrows():
            date_index = row_index[0]
            symbol_index = row_index[1]
            value = row[self.sort_factor_name1]
            if date_index == last_day or last_day == None:
                one_day_data[symbol_index] = value
            else:
                #one_day_data.items().sort(key=lambda x: x[1])
                selected_group_data_list = self.sort_group(one_day_data, self.groups_num, self.group_no)
                self.append(selected_group_data_list, date_index, pred_data)
                one_day_data = {}
            last_day = date_index
        selected_group_data_list = self.sort_group(one_day_data, self.groups_num, self.group_no)
        self.append(selected_group_data_list, last_day, pred_data)
        #print(pred_data)

        preds = pd.DataFrame(data=pred_data, columns=['datetime', 'instrument', 'pred'])
        preds.set_index(['datetime', 'instrument'], inplace=True)
        return preds

    def sort_group(self, data, groups_num, group_selected):
        data_list = sorted(data.items(), key=lambda x: x[1])
        count_per_group = int(len(data) / groups_num)
        start = count_per_group * group_selected
        end = start + count_per_group
        return data_list[start: end]

    def append(self, data, date_index, pred_data):
        for one_data in data:
            pred_data.append((date_index,) + one_data)
            #print(data, date_index)
            #print(pred_data)

class SortStrategy(BaseSignalStrategy):
    def __init__(self, **kwargs):
        print("ChangeStrategy __init__")
        print(kwargs)
        #super(ChangeStrategy, self).__init__(**kwargs)
        super().__init__(**kwargs)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        #print("===============================")
        print("trade_step:", trade_step)
        Logger.analysis("trade_step: {}".format(trade_step))
        #print("trade_calendar:", self.trade_calendar)
        #print("trade_position:", self.trade_position)
        #print("pred_score:", pred_score)
        if pred_score is None:
            return TradeDecisionWO([], self)
        if trade_step % 21 != 0:
            return TradeDecisionWO([], self)

        pred_symbols = pred_score.keys().values
        #print("pred_symbols:", pred_symbols)

        current_temp: Position = copy.deepcopy(self.trade_position)
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        #print("current_stock_list: ", current_stock_list)

        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []
        if len(current_stock_list) > 0:
            sell_symbols = list(set(current_stock_list).difference(set(pred_symbols)))
            Logger.analysis("sell_symbols: {}".format(sell_symbols))
            if len(sell_symbols) > 0:
                for one_symbol in sell_symbols:
                    sell_amount = current_temp.get_stock_amount(code=one_symbol)
                    sell_order = Order(
                        stock_id=one_symbol,
                        amount=sell_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.SELL,  # 0 for sell, 1 for buy
                    )
                    # is order executable
                    if self.trade_exchange.check_order(sell_order):
                        sell_order_list.append(sell_order)
                        trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                            sell_order, position=current_temp
                        )
                        # update cash
                        cash += trade_val - trade_cost

        buy_symbols = list(set(pred_symbols).difference(set(current_stock_list)))
        #print("buy_symbols: ", buy_symbols)
        Logger.analysis("buy_symbols: {}".format(buy_symbols))
        if len(buy_symbols) > 0:
            for one_symbol in buy_symbols:
                # check is stock suspended
                if self.trade_exchange.is_stock_tradable(
                        stock_id=one_symbol,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.BUY,
                ):
                    buy_price = self.trade_exchange.get_deal_price(
                        stock_id=one_symbol, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
                    )
                    #print("buy_price: ", buy_price)
                    buy_amount = int(cash / len(buy_symbols) / buy_price)
                    Logger.analysis("buy_amount: {}".format(buy_amount))
                    factor = self.trade_exchange.get_factor(stock_id=one_symbol, start_time=trade_start_time,
                                                            end_time=trade_end_time)
                    buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
                    buy_order = Order(
                        stock_id=one_symbol,
                        amount=buy_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.BUY,  # 1 for buy
                    )
                    buy_order_list.append(buy_order)
            #print(sell_order_list + buy_order_list)
            return TradeDecisionWO(sell_order_list + buy_order_list, self)


def demonstration(group_no):
    provider_uri = '/Users/zhangyunsheng/Dev/sonata/data/myqlib'  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    feature = SortFeature(instruments=INSTRUMENTS, start_time=SEGMENTS["train"][0], end_time=SEGMENTS['test'][1],
                            fit_start_time=SEGMENTS["train"][0], fit_end_time=SEGMENTS["valid"][1])
    ds = DatasetH(handler=feature,
                  step_len=40,
                  segments=SEGMENTS)
    model = SortModel(group_no)
    backtest_config = {
        "start_time": SEGMENTS["test"][0],
        "end_time": SEGMENTS["test"][1],
        "account": 100000000,
        "benchmark": BENCHMARK,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0002,
            "close_cost": 0.0002,
            "min_cost": 5,
        },
    }

    strategy_obj = SortStrategy(signal=[model, ds])
    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }
    # executor object
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
    # backtest
    portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj,
                                                     **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse("day"))
    #print("Freq.parse(\"day\"):", Freq.parse("day"))
    #print("analysis_freq:", analysis_freq)
    # backtest info
    report_normal_df, positions_normal = portfolio_metric_dict.get(analysis_freq)
    #report_normal_df.to_csv("%sreport_normal_%d" % (REPORT_PATH, GROUP_NO))
    report_normal_df.to_csv(get_report_path(GROUP_NO))
    dump_positions(positions_normal, get_positions_path(GROUP_NO))
    #print("report_normal_df:", report_normal_df)
    #print("positions_normal:", positions_normal)
    #print("qcr.GRAPH_NAME_LIST:", qcr.GRAPH_NAME_LIST)
    fig_list = analysis_position.report_graph(report_normal_df, show_notebook=False)
    for fig in fig_list:
        fig.show()

def group_demonstration():
    global GROUP_NO

    #provider_uri = '/Users/zhangyunsheng/Dev/sonata/data/myqlib'  # target_dir
    #qlib.init(provider_uri=provider_uri, region=REG_CN)
    #feature = SortFeature(instruments=INSTRUMENTS, start_time=SEGMENTS["train"][0], end_time=SEGMENTS['test'][1],
    #                        fit_start_time=SEGMENTS["train"][0], fit_end_time=SEGMENTS["valid"][1])
    #ds = DatasetH(handler=feature,
    #              step_len=40,
    #              segments=SEGMENTS)
    #demonstration(GROUP_NO)
    #return


    UT.check_dir(REPORT_PATH)
    processes = []
    #for i in range(GROUP_NUM):
    #    GROUP_NO = i
    #    print("start group[%d] " % GROUP_NO)
    #    #demonstration(ds, GROUP_NO)
    #    p = multiprocessing.Process(target=demonstration, args=(ds, i))
    #    p.start()
    #    processes.append(p)
    #for p in processes:
    #    p.join()

    pool = multiprocessing.Pool(PROCESS_NUM)
    for i in range(GROUP_NUM):
        GROUP_NO = i
        print("start group[{}] ".format(GROUP_NO))
        Logger.analysis("start group[{}] ".format(GROUP_NO))
        #p.apply(demonstration, args=(ds, i))
        pool.apply_async(demonstration, args=(i,))
    pool.close()
    pool.join()

    rank_correclation()


def rank_correclation():
    #df = pd.DataFrame(data={'A': [1, 2, 8, 3, 4, 5, 6, 7], 'B': [0, 1, 2, 3, 4, 5, 6, 7]})
    #print(df)
    #print(df.corr("spearman"))
    #print(df.corr("kendall"))

    ex_returns_wo_cost = []
    ex_returns_w_cost = []
    return_w_cost = []
    max_drawdown = []
    information_ratio = []
    mean = []
    std = []
    for i in range(GROUP_NUM):
        report_normal = pd.read_csv("%sreport_normal_%d" % (REPORT_PATH, i))
        ex_return_wo_cost_df = risk_analysis(report_normal['return'] - report_normal['bench'])
        ex_return_w_cost_df = risk_analysis(report_normal['return'] - report_normal['bench'] - report_normal['cost'])
        return_w_cost_df = risk_analysis(report_normal['return'] - report_normal['bench'] - report_normal['cost'])
        ex_returns_wo_cost.append(ex_return_wo_cost_df.loc['annualized_return', 'risk'])
        ex_returns_w_cost.append(ex_return_w_cost_df.loc['annualized_return', 'risk'])
        return_w_cost.append(return_w_cost_df.loc['annualized_return', 'risk'])
        max_drawdown.append(return_w_cost_df.loc['max_drawdown', 'risk'])
        information_ratio.append(return_w_cost_df.loc['information_ratio', 'risk'])
        mean.append(return_w_cost_df.loc['mean', 'risk'])
        std.append(return_w_cost_df.loc['std', 'risk'])

    analysis_df = pd.DataFrame(data={
        'group': range(GROUP_NUM),
        'ex_returns_woc': ex_returns_wo_cost,
        'ex_returns_wc': ex_returns_w_cost,
        'return_wc': return_w_cost,
        'max_drawdown': max_drawdown,
        'information_ratio': information_ratio,
        'mean': mean,
        'std': std
    })
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print('analysis_df:\n', analysis_df)
    print('pearson:\n', analysis_df.corr())
    print('spearman:\n', analysis_df.corr('spearman'))
    print('kendall:\n', analysis_df.corr('kendall'))


def show_fig(group_no):
    #report_normal = pd.read_csv("%sreport_normal_%d" % (REPORT_PATH, group_no), parse_dates=True, index_col=0)
    report_normal = pd.read_csv(get_report_path(group_no), parse_dates=True, index_col=0)
    print(report_normal)
    # fig
    fig_list = analysis_position.report_graph(report_normal, show_notebook=False)
    for fig in fig_list:
        fig.show()

def show_fig_all_groups():
    for i in range(GROUP_NUM):
        show_fig(i)

def dump_positions(positions_normal, file_path):
    #positions_data = json.loads(positions_normal)
    positions_data = OrderedDict()
    for key_date, position in positions_normal.items():
        position_data = {}
        position_data['now_account_value'] = position.calculate_value()
        position_data['cash'] = position.get_cash()
        position_data['stock_amount'] = position.get_stock_amount_dict()
        position_data['stock_weight'] = position.get_stock_weight_dict()
        positions_data[DT.date_to_str(key_date)] = position_data
    with open(file_path, 'w') as f:
        json.dump(positions_data, f, indent='  ')

def get_report_path(group_no):
    return "%sreport_normal_%d" % (REPORT_PATH, group_no)

def get_positions_path(group_no):
    return "%spositions_normal_%d" % (REPORT_PATH, group_no)

if __name__ == '__main__':
    group_demonstration()

    #rank_correclation()

    #show_fig(GROUP_NO)
    #show_fig_all_groups()

    #jsonData = '{"a":1,"b":2,"c":3,"d":4,"e":5}'
    #dump_positions(jsonData, get_positions_path(0))
