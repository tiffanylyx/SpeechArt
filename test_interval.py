import time
from direct.showbase.ShowBase import ShowBase
import os

import schedule
import time

def job():
    global count
    print("I'm working...")
    print(count)
    count = count+1
    base.run()

from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
# 输出时间


base = ShowBase()
def printHello():
 print ("Hello")
 print("当前时间戳是", time.time())
 base.run()
 return

def kill():
    print("kill")
    os.kill()
def loop_func(func1, func2, second):
 # 每隔second秒执行func函数
     while True:
      func1()
      func2()
      time.sleep(second)
'''
schedule.every(1).seconds.do(printHello)
count = 0
while True:
    schedule.run_pending()

def job():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
'''
# BlockingScheduler
count = 0
sched = BlockingScheduler()
sched.add_job(job, 'interval', seconds=5, id='my_job_id')
sched.start()
