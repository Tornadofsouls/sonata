#!/usr/bin/python
#-*- coding: utf-8 -*- 

import os
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import utils.const as CT
from utils.logger import Logger
import yaml
#from strategy import *
from strategy.job import Job

def execute(conf):
    job = Job(conf)
    job.execute()

    return 0

def add_job(scheduler, jobs):
    for job in jobs:
        if job['switch'] != 1:
            continue
        second = '*'
        minute = '*'
        hour = '*'
        day = '*'
        month = '*'
        year = '*'
        day_of_week = '*'

        cron = job['cron']
        second = cron['second'] if 'second' in cron else '*'
        minute = cron['minute'] if 'minute' in cron else '*'
        hour = cron['hour'] if 'hour' in cron else '*'
        day = cron['day'] if 'day' in cron else '*'
        month = cron['month'] if 'month' in cron else '*'
        year = cron['year'] if 'year' in cron else '*'
        day_of_week = cron['day_of_week'] if 'day_of_week' in cron else '*'

        scheduler.add_job(execute, 'cron', second=second, minute=minute, hour=hour, day=day, month=month, year=year, day_of_week=day_of_week, args = [job])

    return 0


def main(argv):
    #设置当前工作目录
    os.chdir(CT.HOME)
    Logger.get_instance()

    jobstores = {'default': MemoryJobStore()}
    executors = {
        'default': ThreadPoolExecutor(10),
        #'processpool': ProcessPoolExecutor(3)
    }
    job_defaults = {
        'coalesce': False,
        'max_instances': 3
    }

    scheduler = BlockingScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults)
    #jobs = yaml.load(file(CT.CONF_DIR + 'jobs.yaml'))
    with open(CT.CONF_DIR + 'jobs.yaml', encoding='utf-8') as f:
        jobs = yaml.safe_load(f)
    add_job(scheduler, jobs)
    scheduler.start()

    return

if __name__ == "__main__":
    main(sys.argv)
