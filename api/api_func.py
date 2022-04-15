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

from ctpn import detect
from densenet import ocr, parse
from locard import predict as locard
from vggbctc.predict import single_recognition
from vggbctc.Image import Image
from .utils import check_rotated, helper
from api import logger

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


# 银行卡号识别
def ocr_bank_card(request_id, b64_data):
    #start_time = datetime.now()

    # base64 图片 转为 opencv 数据
    im = __load_image_b64(b64_data)


    # 定位卡号位置 -- opencv
    img = Image(im)

    # 定位卡号位置 -- ctpn
    #_, cropImg, _ = detect.process_one2(img.number_area)
    _, cropImg, box, (im_h, im_w) = detect.process_one2(img.number_area, adjust=False)

    if (box[2] - box[0] < im_w/2): # 框宽度小于图片一半，说明定位有问题，使用ctpn对全图定位
        # 定位卡号位置 -- ctpn
        _, cropImg, _, _ = detect.process_one2(img.img, adjust=False)


    # 保存请求图片(原始的)
    save_backlog('image', request_id, img.img)


    # 清理TF session
    #K.clear_session()
    #tf.compat.v1.reset_default_graph()

    # 识别卡号
    card_number = single_recognition(cropImg)


    # 保存结果
    save_backlog('text', request_id, card_number)


    #print('[Time taken: {!s}]'.format(datetime.now() - start_time))
    return card_number



# 身份证信息识别
def ocr_id_card(request_id, b64_data):
    #start_time = datetime.now()

    # base64 图片 转为 opencv 数据
    #im = __load_image_b64(b64_data)
    im = __load_image_b64(b64_data, remove_color=False, max_size=5000)

    # 保存请求图片(原始的)
    save_backlog('image', request_id, im)

    # 准备 locard 输入
    inputs, h, w = locard.read_img(im)

    # locard定位， box1 是外框， box2 是头像框
    box1, box2 = locard.predict(inputs, h, w)

    #print(box1, box2)

    # 计算需选择角度
    rotate_angle = 0
    if (box1[2]-box1[0]) < (box1[3]-box1[1]): # 高大于宽，说明是竖着的
        rotate_angle = 90

    # 计算box1 box2 的中心
    box1_c = [ (box1[2]-box1[0])/2+box1[0], (box1[3]-box1[1])/2+box1[1] ]
    box2_c = [ (box2[2]-box2[0])/2+box2[0], (box2[3]-box2[1])/2+box2[1] ]

    #print(box1_c, box2_c)

    # 判断照片的位置
    if rotate_angle==0: # 证件 横向
        if box1_c[0] > box2_c[0]: # 照片在左
            rotate_angle = 180
    else: # 证件 竖向
        if box1_c[1] < box2_c[1]: # 照片在下
            rotate_angle = 270
        else:
            rotate_angle = 90

    # 是否需要旋转
    logger.info("rotating %d"%rotate_angle)

    # 截取 证件
    im_card = im[box1[1]:box1[3], box1[0]:box1[2]]
    im_card = check_rotated.rotate_bound(im_card, rotate_angle)

    # 定位文字位置 -- ctpn
    _, boxes = detect.process_text(im_card)

    # 保存请求图片(调整的)
    #if SAVE_IMAGE:
    #    cv2.imwrite(os.path.join(SAVE_IMAGE_PATH, 'id_%s_1.jpg'%request_id), box_im)

    # OCR 文字识别
    r1 = ocr.model(im_card, boxes, True)

    logger.info(r1)

    # 解析身份证信息
    rr1 = parse.parsing_txt(r1)

    # 保存结果
    save_backlog('text', request_id, json.dumps(rr1, ensure_ascii=False))

    #print('[Time taken: {!s}]'.format(datetime.now() - start_time))
    return rr1


# 一般OCR
def ocr_text(request_id, b64_data):
    #start_time = datetime.now()

    # base64 图片 转为 opencv 数据
    im = __load_image_b64(b64_data)
    #im = __load_image_b64(b64_data, remove_color=False)

    # 保存请求图片(原始的)
    save_backlog('image', request_id, im)

    # 定位 -- ctpn
    _, boxes = detect.process_text(im)

    # 保存请求图片(调整的)
    #if SAVE_IMAGE:
    #    cv2.imwrite(os.path.join(SAVE_IMAGE_PATH, 'id_%s_1.jpg'%request_id), im)

    boxes2 = boxes.copy()

    # OCR 文字识别
    rr = ocr.model(im, boxes, True)

    result = []
    for key in sorted(rr.keys()):
        result.append({
            'pos'  : (rr[key][0].tolist())[:8],
            'text' : rr[key][1]
        })

    logger.info(result)

    # 保存结果
    save_backlog('text', request_id, json.dumps(result, ensure_ascii=False))

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
        print( ocr_id_card('warmup', b64_data))
