# coding=utf-8

import os, sys, glob
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
import numpy as np 
import cv2
from model_cnn import get_model
from datetime import datetime
from tqdm import tqdm

input_size = (256,256,3)

json_path = '../data/onebox/json'

output_path = './data/test'

if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir(f"{output_path}/nul")
    os.mkdir(f"{output_path}/fal")
    os.mkdir(f"{output_path}/neg")
    os.mkdir(f"{output_path}/pos")
    os.mkdir(f"{output_path}/non")

#model = get_model('mobile', input_size=input_size, weights=None)
#model.load_weights("./locate_mobile_b128_e30_71_0.98154.h5")

model = get_model('vgg16', input_size=input_size, weights=None)
model.load_weights("../ckpt/locate_onebox_vgg16_b128_e30_157_0.90929.h5")


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

    if j['shapes'][0]['label']=='box':
        p1 = j['shapes'][0]['points']
    else:
        print('label err! ', i)
        return None

    y = np.array([
        p1[0][0]*ratio_x, # card
        p1[0][1]*ratio_y,
        p1[1][0]*ratio_x,
        p1[1][1]*ratio_y,
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

def draw_box(test_path, p1):
    img = cv2.imread(test_path)

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

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x1==x2 or y1==y2:
        return False

    # 截图 box
    crop_img = img[y1:y2, x1:x2].copy()

    #print(rotate_angle)

    # 旋转
    crop_img = rotate_bound(crop_img, rotate_angle)

    basename = os.path.basename(test_path)
    if '_' in basename:
        label = basename.split('_')[2][:3] # 格式 prefix_num_label.jpg
    else:
        label = ''
    cv2.imwrite(f'{output_path}/{label}/crop_{basename}', crop_img)

    # 画框
    cv2.polylines(img, [np.array([ [p1[0], p1[1]], [p1[2], p1[1]], [p1[2], p1[3]], [p1[0], p1[3]] ], np.int32)], 
        True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(f'{output_path}/test_{basename}', img)

    return True


def predict(inputs, h, w): # h,w 为原始图片的 尺寸
    start_time = datetime.now()
    results = model.predict(inputs)
    #print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    p1 = (
        results[0][0]*w,
        results[0][1]*h,
        results[0][2]*w,
        results[0][3]*h,
    )

    return p1, results


def IoU(y_true, y_pred):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]

    w1 = abs(y_true[2] - y_true[0])
    w2 = abs(y_pred[2] - y_pred[0])

    h1 = abs(y_true[3] - y_true[1])
    h2 = abs(y_pred[3] - y_pred[1])

    dw1 = abs(y_true[0] - y_pred[0])
    dw2 = abs(y_true[2] - y_pred[2])

    dh1 = abs(y_true[1] - y_pred[1])
    dh2 = abs(y_true[3] - y_pred[3])

    w3 = (w1 + w2 - dw1 - dw2) / 2
    h3 = (h1 + h2 - dh1 - dh2) / 2

    # intersection area
    intersection = max(w3, 0) * max(h3, 0)

    # area of union of both boxes
    union = w1 * h1 + w2 * h2 - intersection
    
    # iou calculation
    iou = intersection / union

    return iou 


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python %s <img_path>"%sys.argv[0])
        sys.exit(2)

    if os.path.isdir(sys.argv[1]):
        file_list = glob.glob(sys.argv[1]+'/*')
    else:
        file_list = [sys.argv[1]]

    compu_iou = len(file_list)==1

    for ff in tqdm(file_list):
        if os.path.isdir(ff):
            continue
        inputs, h, w = read_img(ff, target_size=input_size[:2])
        p1, pred = predict(inputs, h, w)
        if not draw_box(ff, p1): # 没有试剂盒
            print("Nothing found!")

        if compu_iou:
            # 计算IoU
            truth = read_json(ff)
            if truth is not None:
                print(truth)
                print(pred)
                print(ff, 'IoU = ', IoU(truth, pred[0]))
