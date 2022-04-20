import numpy as np
#from keras.applications import ResNet50
from keras.models import Model
from keras.layers import *


def get_model(input_size = (224,224,3)):
    input=Input(input_size)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    conv1 =BatchNormalization()(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2=BatchNormalization()(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3=BatchNormalization()(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv4=BatchNormalization()(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)

    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv5=BatchNormalization()(conv5)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    # h = Reshape([224 * 224, 3])(conv5)
    # h=Permute([2,1])(h)
    # h=Activation('softmax')(h)
    # out=Reshape([224,224,3])(h)
    #out = Conv2D(3, (1, 1), padding='same',activation='softmax')(conv5)

    bboxHead = Flatten()(conv5)

    #bboxHead = Dense(1024, activation="relu")(bboxHead)
    #bboxHead = Dense(256, activation="relu")(bboxHead)
    bboxHead = Dense(64, activation="relu")(bboxHead)

    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    model = Model(inputs=input, outputs=bboxHead)

    return model