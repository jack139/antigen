# coding=utf-8

import os, sys
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
import numpy as np 
import cv2
from datetime import datetime

from keras.applications import VGG16
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model


input_size = (128,128,3) 

id2label = {0 : 'fal', 1: 'neg', 2 : 'nul', 3 : 'pos'}

# create the base pre-trained model
base_model = VGG16(weights=None, input_shape=input_size, include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
#x = Dropout(0.2)(x)
#x = Dense(32, activation='relu')(x)
# and a logistic layer 
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights("../ckpt/detpos_onebox_vgg16_b512_e10_18_0.99898.h5")

def read_img(test_path,target_size = (224,224)):
    img = cv2.imread(test_path)
    h, w = img.shape[:2]
    img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
    #img = img / 255.
    img = np.reshape(img,(1,)+img.shape)
    return img, h, w


def predict(inputs): # h,w 为原始图片的 尺寸
    start_time = datetime.now()
    results = model.predict(inputs)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    return results, id2label[results.argmax()]


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python %s <img_path>"%sys.argv[0])
        sys.exit(2)

    inputs, h, w = read_img(sys.argv[1], target_size=input_size[:2])
    pred = predict(inputs)

    print(pred)
