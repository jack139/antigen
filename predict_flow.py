# coding=utf-8

import os, sys, glob
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
import numpy as np 
import cv2
from locate.model import get_model
from datetime import datetime
from tqdm import tqdm

from keras.applications import VGG16
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

import tensorflow as tf
from config.settings import DETPOS_WEIGHTS, LOCATE_WEIGHTS, GPU_MEMORY_LOCATE, GPU_RUN_LOCATE

# GPU内存控制
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_LOCATE)

# 是否强制使用 CPU
if GPU_RUN_LOCATE:
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
else:
    config = tf.ConfigProto(device_count = {'CPU' : 1, 'GPU' : 0}, gpu_options=gpu_options)

# 建立默认session
graph = tf.Graph()  # 解决多线程不同模型时，keras或tensorflow冲突的问题
session = tf.Session(graph=graph, config=config)

locate_input_size = (256,256,3)
detpos_input_size = (128,128,3) 
id2label = {0 : 'fal', 1: 'neg', 2 : 'nul', 3 : 'pos'}

with graph.as_default():
    with session.as_default():
        # 定位模型
        locate_model = get_model('vgg16', input_size=locate_input_size, weights=None)
        locate_model.load_weights(LOCATE_WEIGHTS)
        print('Lcate model load_weights: ', LOCATE_WEIGHTS)

        # 识别模型
        base_model = VGG16(weights=None, input_shape=detpos_input_size, include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(4, activation='softmax')(x)
        detpos_model = Model(inputs=base_model.input, outputs=predictions)
        detpos_model.load_weights(DETPOS_WEIGHTS)
        print('Detpos model load_weights: ', DETPOS_WEIGHTS)


def read_img(test_path,target_size = (224,224)):
    img = cv2.imread(test_path)
    h, w = img.shape[:2]
    img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
    img = img / 255.
    img = np.reshape(img,(1,)+img.shape)
    return img, h, w

def rotate_bound(image,angle):
    #获取图像的尺寸
    #旋转中心
    (h,w) = image.shape[:2]
    (cx,cy) = (w/2,h/2)
    
    #设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    # 计算图像旋转后的新边界
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    
    return cv2.warpAffine(image,M,(nW,nH))


def crop_box(img, p1):

    # 计算需选择角度
    rotate_angle = 0
    box1 = p1

    if box1[0]<box1[2]: # 起点 在左
        if box1[1]<box1[3]: # 起点 在上
            rotate_angle = 0
            x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
        else:
            rotate_angle = 90
            x1, y1, x2, y2 = box1[0], box1[3], box1[2], box1[1]
    else: # 起点 在右
        if box1[1]<box1[3]: # 起点 在上
            rotate_angle = 270
            x1, y1, x2, y2 = box1[2], box1[1], box1[0], box1[3]
        else:
            rotate_angle = 180
            x1, y1, x2, y2 = box1[2], box1[3], box1[0], box1[1]

    # 截图 box
    crop_img = img[int(y1):int(y2), int(x1):int(x2)].copy()

    #print(rotate_angle)

    # 旋转
    crop_img = rotate_bound(crop_img, rotate_angle)

    return crop_img


def locate_predict(inputs, h, w): # h,w 为原始图片的 尺寸
    start_time = datetime.now()
    with graph.as_default(): # 解决多线程不同模型时，keras或tensorflow冲突的问题
        with session.as_default():
            results = locate_model.predict(inputs)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    p1 = (
        results[0][0]*w,
        results[0][1]*h,
        results[0][2]*w,
        results[0][3]*h,
    )

    return p1, results


def detpos_predict(inputs): 
    start_time = datetime.now()
    with graph.as_default(): # 解决多线程不同模型时，keras或tensorflow冲突的问题
        with session.as_default():
            results = detpos_model.predict(inputs)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    return results, id2label[results.argmax()]


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python %s <img_path>"%sys.argv[0])
        sys.exit(2)

    inputs, h, w = read_img(sys.argv[1], target_size=locate_input_size[:2])
    p1, pred = locate_predict(inputs, h, w)
    if pred.sum()<1e-2: # 没有试剂盒
        print("Nothing found!")
    else:
        img = cv2.imread(sys.argv[1])
        crop_img = crop_box(img, p1)
        crop_img = cv2.resize(crop_img, detpos_input_size[:2], interpolation = cv2.INTER_AREA)
        crop_img = np.reshape(crop_img,(1,)+crop_img.shape)
        detpos_pred = detpos_predict(crop_img)
        print(detpos_pred)
