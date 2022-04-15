# coding:utf-8

import os
from flask import Flask, Blueprint, render_template, request
import urllib3, json, base64, time, hashlib
from datetime import datetime
from api.utils import helper
from config.settings import MAX_IMAGE_SIZE
from .. import logger

logger = logger.get_logger(__name__)


demo_app = Blueprint('demo', __name__)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



# 接口演示
@demo_app.route("/demo", methods=["GET"])
def demo_get():
    return render_template('demo.html')

@demo_app.route("/demo", methods=["POST"])
def demo_post():
    file = request.files['file']
    cate = request.form['cate']
    if file and allowed_file(file.filename):
        #file.save(os.path.join(os.getcwd(), file.filename))
        api_url, params, status, rdata, timespan = call_api(cate, file.stream.read())
        return render_template('result.html', 
            result=rdata, status=status, 
            timespan=timespan, params=params, api_url=api_url)
    else:
        return "not allowed image"


# 调用接口
def call_api(cate, img_data):
    hostname = '127.0.0.1'

    body = {
        'version'  : '1',
        'signType' : 'SHA256', 
        #'signType' : 'SM2',
        'encType'  : 'plain',
        'data'     : {
            'image'    : base64.b64encode(img_data).decode('utf-8'),
        }
    }

    appid = '66A095861BAE55F8735199DBC45D3E8E'
    unixtime = int(time.time())
    body['timestamp'] = unixtime
    body['appId'] = appid

    param_str = helper.gen_param_str(body)
    sign_str = '%s&key=%s' % (param_str, '43E554621FF7BF4756F8C1ADF17F209C')

    if body['signType'] == 'SHA256':
        signature_str =  base64.b64encode(hashlib.sha256(sign_str.encode('utf-8')).hexdigest().encode('utf-8')).decode('utf-8')
    else: # SM2
        signature_str = sm2.SM2withSM3_sign_base64(sign_str)

    #print(sign_str)

    body['signData'] = signature_str

    body_str = json.dumps(body)
    #print(body)

    pool = urllib3.PoolManager(num_pools=2, timeout=180, retries=False)

    host = 'http://%s:5000'%hostname
    url = host+'/antigen/check'

    start_time = datetime.now()
    r = pool.urlopen('POST', url, body=body_str)
    #print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    print(r.status)
    if r.status==200:
        rdata = json.dumps(json.loads(r.data.decode('utf-8')), ensure_ascii=False, indent=4)
    else:
        rdata = r.data


    body['data']['image'] = body['data']['image'][:20]+' ... ' + body['data']['image'][-20:]
    body2 = json.dumps(body, ensure_ascii=False, indent=4)
    return url, body2, r.status, rdata, \
        '{!s}s'.format(datetime.now() - start_time)
