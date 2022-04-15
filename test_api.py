# coding:utf-8
import sys
import urllib3, json, base64, time, hashlib
from datetime import datetime
from api.utils import helper, sm2

urllib3.disable_warnings()


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python3 %s <host> <image_path>" % sys.argv[0])
        sys.exit(2)

    hostname = sys.argv[1]
    filepath = sys.argv[2]

    with open(filepath, 'rb') as f:
        img_data = f.read()

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

    body = json.dumps(body)
    #print(body)

    pool = urllib3.PoolManager(num_pools=2, timeout=180, retries=False)

    host = 'http://%s:5000'%hostname
    url = host+'/antigen/check'

    start_time = datetime.now()
    r = pool.urlopen('POST', url, body=body)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    print(r.status)
    if r.status==200:
        print(json.loads(r.data.decode('utf-8')))
    else:
        print(r.data)
