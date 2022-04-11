# coding=utf-8

import os, sys
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
import numpy as np 
import cv2
from model import get_model
from datetime import datetime

input_size = (224,224,3)

json_path = '../data/json'

model = get_model('vgg16', weights=None)
model.load_weights("locate_vgg16_b16_e10_20_0.82327.h5")
#model = get_model('densenet', weights=None)
#model.load_weights('locard_densenet_b32_e10_100_0.82439.h5')

def read_img(test_path,target_size = (224,224)):
    img = cv2.imread(test_path)
    h, w = img.shape[:2]
    img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
    img = img / 255.
    img = np.reshape(img,(1,)+img.shape)
    return img, h, w

def read_json(test_path):
    # 准备标记
    file_name = os.path.split(test_path)[-1]
    json_file = os.path.join(json_path, os.path.splitext(file_name)[0]+'.json')
    if not os.path.exists(json_file):
        return None

    with open(json_file) as fp:
        j = json.load(fp)

    ratio_x = 1.0 / j['imageWidth']
    ratio_y = 1.0 / j['imageHeight']

    if j['shapes'][0]['label']=='CT' and j['shapes'][1]['label']=='S':
        p1 = j['shapes'][0]['points']
        p2 = j['shapes'][1]['points']
    elif j['shapes'][0]['label']=='S' and j['shapes'][1]['label']=='CT':
        p1 = j['shapes'][1]['points']
        p2 = j['shapes'][0]['points']
    else:
        print('label err! ', i)
        return None

    y = np.array([
        p1[0][0]*ratio_x, # card
        p1[0][1]*ratio_y,
        p1[1][0]*ratio_x,
        p1[1][1]*ratio_y,
        p2[0][0]*ratio_x, # photo
        p2[0][1]*ratio_y,
        p2[1][0]*ratio_x,
        p2[1][1]*ratio_y,                
    ])

    return y


def draw_box(test_path, p1, p2):
    img = cv2.imread(test_path)
    cv2.polylines(img, [np.array([ [p1[0], p1[1]], [p1[2], p1[1]], [p1[2], p1[3]], [p1[0], p1[3]] ], np.int32)], 
        True, color=(0, 255, 0), thickness=2)
    cv2.polylines(img, [np.array([ [p2[0], p2[1]], [p2[2], p2[1]], [p2[2], p2[3]], [p2[0], p2[3]] ], np.int32)], 
        True, color=(0, 255, 0), thickness=2)
    cv2.imwrite('data/test_result.jpg', img)


def predict(inputs, h, w): # h,w 为原始图片的 尺寸
    start_time = datetime.now()
    results = model.predict(inputs)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    p1 = (
        results[0][0]*w,
        results[0][1]*h,
        results[0][2]*w,
        results[0][3]*h,
    )

    p2 = (
        results[0][4]*w,
        results[0][5]*h,
        results[0][6]*w,
        results[0][7]*h,
    )

    print(results)

    return p1, p2, results


# box:[上, 左, 下, 右]
def IoU(box1, box2):
    # 计算中间矩形的宽高
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    
    # 计算交集、并集面积
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    # 计算IoU
    iou = inter / union
    return iou


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python %s <img_path>"%sys.argv[0])
        sys.exit(2)

    inputs, h, w = read_img(sys.argv[1], target_size=input_size[:2])
    p1, p2, pred = predict(inputs, h, w)
    draw_box(sys.argv[1], p1, p2)

    # 计算IoU
    truth = read_json(sys.argv[1])
    if truth is not None:
        print(truth)
        box1 = [
            truth[1],
            truth[0],
            truth[3],
            truth[2],
        ]

        box2 = [
            pred[0][1],
            pred[0][0],
            pred[0][3],
            pred[0][2],
        ]
        print('IoU = ', IoU(box1, box2))
