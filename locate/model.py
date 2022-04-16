# coding=utf-8

import numpy as np
from keras.applications import ResNet50, VGG16, MobileNetV2, DenseNet121
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Flatten, LeakyReLU


def get_model(model_type='vgg16', input_size = (224,224,3), freeze=False, weights='imagenet'):
    # create the base pre-trained model
    if model_type=='resnet':
        base_model = ResNet50(weights=weights, input_shape=input_size, include_top=False)
    elif model_type=='mobile':
        base_model = MobileNetV2(weights=weights, include_top=False, input_tensor=Input(shape=input_size))
    elif model_type=='dense':
        base_model = DenseNet121(weights=weights, include_top=False, input_tensor=Input(shape=input_size))
    else:
        base_model = VGG16(weights=weights, include_top=False, input_tensor=Input(shape=input_size))

    # freeze all VGG layers so they will *not* be updated during the
    # training process
    if freeze:
        print("[INFO] freeze all %s base_model layers..."%model_type)
        for layer in base_model.layers:
            layer.trainable = False

    # flatten the max-pooling output of VGG
    flatten = base_model.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates

    # DenseNet
    #bboxHead = Dense(64, activation="relu")(flatten)
    #bboxHead = Dense(32, activation="relu")(bboxHead)
    #bboxHead = Dense(16, activation="relu")(bboxHead)

    # VGG
    bboxHead = Dense(1024, activation="relu")(flatten)
    bboxHead = Dense(256, activation="relu")(bboxHead)
    bboxHead = Dense(64, activation="relu")(bboxHead)

    #bboxHead = Dense(1024)(flatten)
    #bboxHead = LeakyReLU(alpha=0.02)(bboxHead)
    #bboxHead = Dropout(0.2)(bboxHead)
    #bboxHead = Dense(256)(bboxHead)
    #bboxHead = LeakyReLU(alpha=0.02)(bboxHead)
    #bboxHead = Dropout(0.2)(bboxHead)
    #bboxHead = Dense(64)(bboxHead)
    #bboxHead = LeakyReLU(alpha=0.02)(bboxHead)
    #bboxHead = Dropout(0.2)(bboxHead)

    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=base_model.input, outputs=bboxHead)

    return model
