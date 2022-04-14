# coding=utf-8

import os, sys, glob
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
import numpy as np 
import cv2
from model import get_model
from datetime import datetime

input_size = (256,256,3)

json_path = '../data/json'

model = get_model('vgg16', input_size=input_size, weights=None)
model.load_weights("../ckpt/locate_vgg16_b16_e10_20_0.82327.h5")


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

    if j['shapes'][0]['label']=='box' and j['shapes'][1]['label']=='CT':
        p1 = j['shapes'][0]['points']
        p2 = j['shapes'][1]['points']
    elif j['shapes'][0]['label']=='CT' and j['shapes'][1]['label']=='box':
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

def draw_box(test_path, p1, p2):
    img = cv2.imread(test_path)

    # 截图
    if p1[0]<p1[2]:
        x1, x2 = p1[0], p1[2]
    else:
        x1, x2 = p1[2], p1[0]
    if p1[1]<p1[3]:
        y1, y2 = p1[1], p1[3]
    else:
        y1, y2 = p1[3], p1[1]
    crop_img = img[int(y1):int(y2), int(x1):int(x2)].copy()

    # 计算需选择角度
    rotate_angle = 0
    box1, box2 = p1, p2
    if (box1[2]-box1[0]) < (box1[3]-box1[1]): # 高大于宽，说明是竖着的
        rotate_angle = 90

    # 计算box1 box2 的中心
    box1_c = [ (box1[2]-box1[0])/2+box1[0], (box1[3]-box1[1])/2+box1[1] ]
    box2_c = [ (box2[2]-box2[0])/2+box2[0], (box2[3]-box2[1])/2+box2[1] ]

    # 判断CT的位置
    if rotate_angle==0: # 横向
        if box1_c[0] > box2_c[0]: # CT在左
            rotate_angle = 180
    else: # 证件 竖向
        if box1_c[1] < box2_c[1]: # CT在下
            rotate_angle = 270
        else:
            rotate_angle = 90

    # 旋转
    crop_img = rotate_bound(crop_img, rotate_angle)

    basename = os.path.basename(test_path)
    cv2.imwrite(f'data/{basename}', crop_img)

    # 划线
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

    return p1, p2, results


def IoU(y_true, y_pred):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]

    # AOG = Area of Groundtruth box
    AoG = abs(y_true[2] - y_true[0] + 1) * abs(y_true[3] - y_true[1] + 1)
    
    # AOP = Area of Predicted box
    AoP = abs(y_pred[2] - y_pred[0] + 1) * abs(y_pred[3] - y_pred[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = max(y_true[0], y_pred[0])
    overlap_1 = max(y_true[1], y_pred[1])
    overlap_2 = min(y_true[2], y_pred[2])
    overlap_3 = min(y_true[3], y_pred[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    return iou 


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python %s <img_path>"%sys.argv[0])
        sys.exit(2)

    if os.path.isdir(sys.argv[1]):
        file_list = glob.glob(sys.argv[1])
    else:
        file_list = [sys.argv[1]]

    for ff in file_list:
        inputs, h, w = read_img(ff, target_size=input_size[:2])
        p1, p2, pred = predict(inputs, h, w)
        draw_box(ff, p1, p2)

        # 计算IoU
        truth = read_json(ff)
        if truth is not None:
            #print(truth)
            #print(pred)
            print('IoU = ', IoU(truth, pred[0]))
