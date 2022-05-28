import os
import sys
import glob
import json
from tqdm import tqdm
import cv2
import numpy as np

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

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if abs(x1-x2)<12 or abs(y1-y2)<12: # 没有结果
        return None

    # 截图 box
    crop_img = img[y1:y2, x1:x2].copy()

    #print(rotate_angle)

    # 旋转
    crop_img = rotate_bound(crop_img, rotate_angle)

    return crop_img


if __name__ == '__main__':
    if len(sys.argv)<3:
        print("usage: python %s <img_path> <json_path>"%sys.argv[0])
        sys.exit(2)

    file_list = glob.glob(sys.argv[1]+'/*')
    json_path = sys.argv[2]

    for ff in tqdm(file_list):
        filepath, basename = os.path.split(ff)
        filename, ext = os.path.splitext(basename)
        json_file = os.path.join(json_path, filename+'.json')

        # 结果目录
        os.makedirs(os.path.join(filepath, 'result'), exist_ok=True)

        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            print('--->', ff, 'is not a IMAGE.')
            continue

        with open(json_file) as fp:
            j = json.load(fp)

        p0 = j['shapes'][0]['points']
        p1 = [ p0[0][0], p0[0][1], p0[1][0], p0[1][1] ]

        img = cv2.imread(ff)
        crop_img0 = crop_box(img, p1)

        if crop_img0 is None:
            print(ff, 'is None')
            continue

        cv2.imwrite(os.path.join(filepath, 'result', f'crop_{basename}'), crop_img0)
