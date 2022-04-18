# -*- coding: utf-8 -*-

import base64
from io import BytesIO
import os
import json
from datetime import datetime

import numpy as np
import cv2

from keras import backend as K
import tensorflow as tf

from .utils import helper
from api import logger

import predict_flow

from config.settings import SAVE_IMAGE, SAVE_IMAGE_PATH, WARM_UP_IMAGES

logger = logger.get_logger(__name__)

# 将 base64 编码的图片转为 opencv 数组
def __load_image_b64(b64_data, remove_color=True, max_size=1500):
    data = base64.b64decode(b64_data) # Bytes
    tmp_buff = BytesIO()
    tmp_buff.write(data)
    tmp_buff.seek(0)
    file_bytes = np.asarray(bytearray(tmp_buff.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if remove_color:
        img = img[:, :, ::-1] # 去色，强化图片
    tmp_buff.close()
    # 压缩处理
    max_width = max(img.shape)
    if max_width>max_size: # 图片最大宽度为 1500
        ratio = max_size/max_width
        img = cv2.resize(img, (round(img.shape[1]*ratio), round(img.shape[0]*ratio)))
    return img


def save_backlog(t, request_id, data):
    if SAVE_IMAGE:
        output_dir = os.path.join(SAVE_IMAGE_PATH, helper.time_str(format=2)[2:])
        os.makedirs(output_dir, exist_ok=True)
        if t=='image':
            cv2.imwrite(os.path.join(output_dir, 'card_%s.jpg'%request_id), data)
        else:
            with open(os.path.join(output_dir, 'card_%s.txt'%request_id), 'w') as f:
                f.write(data)


# 试剂盒识别
def detpos_check(request_id, b64_data):
    #start_time = datetime.now()

    # base64 图片 转为 opencv 数据
    img = __load_image_b64(b64_data, remove_color=False, max_size=5000)
    h, w = img.shape[:2]

    # 保存请求图片(原始的)
    save_backlog('image', request_id, img)

    # 准备定位模型输入
    inputs = cv2.resize(img, predict_flow.locate_input_size[:2], interpolation = cv2.INTER_AREA)
    inputs = inputs / 255.
    inputs = np.reshape(inputs,(1,)+inputs.shape)

    # 定位预测
    p1, pred = predict_flow.locate_predict(inputs, h, w)
    if pred.sum()<1e-2: # 没有试剂盒
        #print("Nothing found!")
        result = "none"
    else:
        crop_img = predict_flow.crop_box(img, p1)
        crop_img = cv2.resize(crop_img, predict_flow.detpos_input_size[:2], interpolation = cv2.INTER_AREA)
        crop_img = np.reshape(crop_img,(1,)+crop_img.shape)
        _, result = predict_flow.detpos_predict(crop_img)

    # 保存结果
    save_backlog('text', request_id, result)

    #print('[Time taken: {!s}]'.format(datetime.now() - start_time))
    return result


# 模型预热
def warm_up():
    file_list = os.listdir(WARM_UP_IMAGES)
    for file in file_list:
        filepath = os.path.join(WARM_UP_IMAGES, file)
        with open(filepath, 'rb') as f:
            img_data = f.read()
        b64_data = base64.b64encode(img_data).decode('utf-8')

        print('warmup >>>>>', filepath)
        print( detpos_check('warmup', b64_data))
