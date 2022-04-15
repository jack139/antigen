# coding:utf-8

import json, time, random
from datetime import datetime
from flask_restful import reqparse, abort, Resource, fields, request
from config.settings import MAX_IMAGE_SIZE
from ..utils import helper
from .. import logger


logger = logger.get_logger(__name__)

# 异步处理压测实验
class StressTest(Resource):
    @helper.signature_required
    def post(self): 

        ret_json = {
            "appId"     : '',
            "code"      : 9000,
            "success"   : False,
            "signType"  : "plain",
            "encType"   : "plain",
            "data"      : {},
            "timestamp" : int(time.time()),
        }

        try:
            # 获取入参
            body_data = request.get_data().decode('utf-8') # bytes to str
            json_data = json.loads(body_data)
 
            ret_json['appId'] = json_data['appId']


            # 准备发队列消息
            request_id = helper.gen_request_id()

            request_msg = {
                'api'   : 'stress_test',
                'delay' : random.random()*0.2 # 随机延时 0-2 秒
            }

            start_time = datetime.now()

            # 异步处理

            # 在发redis消息前注册, 防止消息漏掉
            ps = helper.redis_subscribe(request_id)

            # 发布消息给redis
            r = helper.redis_publish_request(request_id, request_msg)
            if r is None:
                logger.error("消息队列异常")
                ret_json["code"] = 9009
                ret_json["data"] = {"msg": "消息队列异常"}
                return ret_json

            # 通过redis订阅等待结果返回
            ret = helper.redis_sub_receive(ps, request_id)               
            ret2 = json.loads(ret['data'].decode('utf-8'))


            #print('<--', request_id, helper.time_str(), datetime.now() - start_time)
            logger.info('[Time taken: {!s}]'.format(datetime.now() - start_time))
            
            if ret2['code']==200: # 内部成功使用 200
                ret_json["success"] = True
            ret_json["code"] = 0  # 对外成功返回 0
            ret_json["data"] = ret2['data']            
            return ret_json

        except Exception as e:
            logger.error("未知异常: %s" % e, exc_info=True)
            ret_json["code"] = 9999
            ret_json["data"] = {"msg": "%s : %s" % (e.__class__.__name__, e) }
            return ret_json

