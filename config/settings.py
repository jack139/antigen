# -*- coding: utf-8 -*-

############# 算法相关设置

# ctpn 预训练权重的路径
CTPN_CHECKPOINT = '/home/tao/Codes/cv/ocr_model/ctpn/'

# 数字识别模型 权重文件
VGG_B_CTC_WEIGHTS = '/home/tao/Codes/cv/ocr_model/vgg_b_ctc_weight.h5'

# densenet 预训练权重的路径
#DENSENET_WEIGHTS = '/home/tao/Codes/cv/ocr_model/weights_densenet.h5'
DENSENET_WEIGHTS = '/home/tao/Codes/cv/ocr_model/ocr-densenet-01-0.9318-0.0749-0.9848.weights'

# locard 预训练权重的路径
LOCARD_WEIGHTS = '/home/tao/Codes/cv/ocr_model/locard_vgg16_b32_e10_100_0.94010.h5'


# 模型预热图片 路径
WARM_UP_IMAGES = '../warmup'

# GPU 内存设置
GPU_MEMORY_CTPN = 0.2
GPU_MEMORY_DENSENET = 0.00001
GPU_MEMORY_CTC = 0.1
GPU_MEMORY_LOCARD = 0.00001
GPU_RUN_DENSENET = False
GPU_RUN_CTC = False
GPU_RUN_LOCARD = False

############  app server 相关设置

APP_NAME = 'cardnum'
APP_NAME_FORMATED = 'CARDNUM'

# 参数设置
DEBUG_MODE = False
BIND_ADDR = '0.0.0.0'
BIND_PORT = '5000'

# 是否保存请求的图片和结果
SAVE_IMAGE = True
SAVE_IMAGE_PATH = '/tmp/cardnum'

# dispatcher 中 最大线程数
MAX_DISPATCHER_WORKERS = 8


############# appid - appsecret

SECRET_KEY = {
    '19E179E5DC29C05E65B90CDE57A1C7E5' : 'D91CEB11EE62219CD91CEB11EE62219C',
    '66A095861BAE55F8735199DBC45D3E8E' : '43E554621FF7BF4756F8C1ADF17F209C',
    '75C50F018B34AC0240915EC685F5961B' : 'BCB3DF17A794368E1BB0352D3D2D5F50',
    '3EA25569454745D01219080B779F021F' : '41DF0E6AE27B5282C07EF5124642A352'
}


############# 消息中间件设置

REDIS_CONFIG = {
    'SERVER' : '127.0.0.1',
    'PORT'   : '7480',
    'PASSWD' : 'e18ffb7484f4d69c2acb40008471a71c',
    'REQUEST-QUEUE' : 'antigen-synchronous-asynchronous-queue',
    'REQUEST-QUEUE-NUM' : 1,
    'MESSAGE_TIMEOUT' : 10, # 结果返回消息超时，单位：秒
}

# 图片数据最大尺寸
MAX_IMAGE_SIZE = 1024*1024*3  # 3MB
