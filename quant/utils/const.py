#!/usr/bin/python
#-*- coding: utf-8 -*- 
import os

#目录
HOME = os.path.dirname(os.path.abspath(__file__)) + '/../../'
BASICS_DIR = HOME + 'data/basics/'
DAILY_DIR = HOME + 'data/daily/'
HFQ_DIR = HOME + 'data/hfq/'
TICK_DIR = HOME + 'data/tick/'
MINUTE_DIR = HOME + 'data/minute/'
HSGT_DIR = HOME + 'data/hsgt/'
MODEL_DIR = HOME + 'data/model/'
LOG_DIR = HOME + 'log/'
CONF_DIR = HOME + 'conf/'
PORTFOLIO_CONF_DIR = CONF_DIR + 'portfolio/'
MY_QLIB_DIR = HOME + 'data/myqlib/'
MY_QLIB_INSTRUMENTS_DIR = MY_QLIB_DIR + 'instruments/'

# 实时行情数据
SPOT_FILE = 'spot'

# 文件后缀名
FILE_SUFFIX = '.csv'

# csv存储分隔
COLUMN_SEP = ','

#历史数据开始时间
START = '2001-01-01'


#线程数
HIS_THRD_CNT = 10
TICK_THRD_CNT = 10

BROKER_NAME_MANUAL = 'manual'
BROKER_NAME_XUEQIU = 'xueqiu'
BROKER_NAME_YINHE = 'yinhe'
