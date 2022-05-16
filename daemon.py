# -*- coding: utf-8 -*-
#
# 后台daemon进程，启动后台处理进程，并检查进程监控状态
#

import sys
import time, shutil, os
from api.utils import helper
from config.settings import LOCAL_DISPATCHER_QUEUE_ID


APP_DIR=''
LOG_DIR=''

def start_processor(pname, param=''):
    cmd_path = '%s/%s.pyc'%(APP_DIR, pname)
    if not os.path.exists(cmd_path):
        cmd_path = '%s/%s.py'%(APP_DIR, pname)
    cmd0="nohup python3 %s %s >> %s/antigen_%s.log 2>&1 &" % \
        (cmd_path, param, LOG_DIR, pname+param.replace(' ',''))
    print('start process: ', cmd0)
    os.system(cmd0)

def get_processor_pid(pname):
    cmd0='pgrep -f "%s"' % pname
    pid=os.popen(cmd0).readlines()
    if len(pid)>0:
        return pid[0].strip()
    else:
        return None

def kill_processor(pname):
    cmd0='kill -9 `pgrep -f "%s"`' % pname
    os.system(cmd0)
    time.sleep(1)

if __name__=='__main__':
    if len(sys.argv)<3:
        print("usage: daemon.py <APP_DIR> <LOG_DIR>")
        sys.exit(2)

    APP_DIR=sys.argv[1]
    LOG_DIR=sys.argv[2]

    print("DAEMON: %s started" % helper.time_str())
    print("APP_DIR=%s\nLOG_DIR=%s" % (APP_DIR, LOG_DIR))

    #
    #启动后台进程
    #
    kill_processor('%s/dispatcher' % APP_DIR)
    for i in LOCAL_DISPATCHER_QUEUE_ID:
        start_processor('dispatcher', str(i))

    try:    
        _count=_ins=0
        while 1:
            try:
                # 检查processor进程 dispatcher
                pid=get_processor_pid('%s/dispatcher' % APP_DIR)
                if pid==None:
                    # 进程已死, 重启进程
                    kill_processor('%s/dispatcher' % APP_DIR)
                    for i in LOCAL_DISPATCHER_QUEUE_ID:
                        start_processor('dispatcher', str(i))
                    _ins+=1
                    print("%s\tdispatcher restart" % helper.time_str())

            except Exception as e:
                print("%s\tException: %s" % (helper.time_str(), e))
                        
            time.sleep(20)
            _count+=1
            if _count>1000:
                if _ins>0:
                    print("%s  HEARTBEAT: error %d" % (helper.time_str(), _ins))
                else:
                    print("%s  HEARTBEAT: fine." % (helper.time_str()))
                _count=_ins=0
            sys.stdout.flush()

    except KeyboardInterrupt:
        print('\nCtrl-C!')
        kill_processor('%s/dispatcher' % APP_DIR)

    print("DAEMON: %s exited" % helper.time_str())
