# coding=utf-8

import os, json
import numpy as np 
import cv2

def dataGenerator(data_path,json_path,batch_size=64,target_size = (224,224)):
    file_list = os.listdir(data_path)
    #file_list = sorted(file_list)

    X_data = []
    y_data = []

    while True:

        np.random.shuffle(file_list)
        for n,i in enumerate(file_list):
            # 准备标记
            json_file = os.path.join(json_path, os.path.splitext(i)[0]+'.json')
            with open(json_file) as fp:
                j = json.load(fp)
            #ratio_x = target_size[0] / j['imageWidth']
            #ratio_y = target_size[1] / j['imageHeight']

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
                continue

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

            # 准备图片
            img = cv2.imread(os.path.join(data_path,i))
            img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)

            #cv2.rectangle(img, (int(y[0]), int(y[1])), (int(y[2]), int(y[3])), (255,0,0), 2)
            #cv2.rectangle(img, (int(y[4]), int(y[5])), (int(y[6]), int(y[7])), (255,0,0), 2)
            #cv2.imwrite(os.path.join("tmp","tmp_%s"%i), img)

            img = img / 255.

            #img = np.reshape(img,(1,)+img.shape) # （1,256,256,3）
            #y = np.reshape(y,(1,8)) # （1, 8, 1）

            # only for test
            #print(i, y)
            #print(img.shape, y.shape)

            #yield (img, y)

            X_data.append(img)
            y_data.append(y)

            if len(X_data) == batch_size or n == len(file_list)-1:
                X_data = np.array(X_data, dtype="float32")
                y_data = np.array(y_data, dtype="float32")

                yield [X_data, y_data]

                X_data = []
                y_data = []

