# coding=utf-8

# PAFPN 1803.01534 + Resnet50 v2

from keras.applications import ResNet50V2
from keras.models import Model
from keras.layers import *


def get_backbone_ResNet50(input_shape, weights):
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = ResNet50V2(include_top=False, input_shape=input_shape, weights=weights)
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return backbone.input, [c3_output, c4_output, c5_output]


def get_model(input_size = (256,256,3), weights='imagenet'):
    base_inputs, base_outputs = get_backbone_ResNet50(input_shape=input_size, weights=weights)
    c3_output, c4_output, c5_output = base_outputs

    p3_output_1 = Conv2D(256, 1, strides=1, activation='relu', padding="same")(c3_output) #(None, 16, 16)
    p4_output_1 = Conv2D(256, 1, strides=1, activation='relu', padding="same")(c4_output) #(None, 8, 8)
    p5_output_1 = Conv2D(256, 1, strides=1, activation='relu', padding="same")(c5_output) #(None, 8, 8)

    # 红线箭头
    p5_output = Add()([UpSampling2D(2)(p5_output_1), p3_output_1]) #(None, 16, 16)

    # p3, p4, p5
    p4_output = UpSampling2D(2)(p4_output_1) #(None, 16, 16)
    p4_output = Add()([p5_output, p4_output]) #(None, 16, 16)
    p3_output = Add()([p4_output, p3_output_1]) #(None, 16, 16)

    # n3, n4, n5
    n3_output = Conv2D(256, 3, strides=1, activation='relu', padding="same")(p3_output)

    n3_output_3 = Conv2D(256, 3, strides=1, activation='relu', padding="same")(n3_output)
    n4_output = Add()([p4_output, n3_output_3])

    n4_output_3 = Conv2D(256, 3, strides=1, activation='relu', padding="same")(n4_output)
    n5_output = Add()([p5_output, n4_output_3])

    # 绿线箭头
    n5_output = Add()([p5_output, p3_output_1])

    n3_output = Flatten()(n3_output)
    n4_output = Flatten()(n4_output)
    n5_output = Flatten()(n5_output)

    m1_output = Concatenate(axis=1)([n3_output,
                                     n4_output,
                                     n5_output])

    #m1_output = Flatten()(m1_output)

    m1_output = Dense(256, activation='relu', kernel_initializer='he_uniform')(m1_output)
    m1_output = Dense(64, activation='relu', kernel_initializer='he_uniform')(m1_output)
    m1_output = Dense(4, activation='sigmoid', kernel_initializer='he_normal')(m1_output)

    model = Model(inputs=base_inputs, outputs=m1_output)

    return model
