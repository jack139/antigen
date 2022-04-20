# coding=utf-8

# Panoptic FPN 1901.02446

from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import *

# Panoptic FPN 1901.02446 + MobileNet v2

def get_backbone_MobileNet(input_shape, weights):
    """Builds MobileNetV2 with pre-trained imagenet weights"""
    backbone = MobileNetV2(include_top=False, input_shape=input_shape, weights=weights)
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in [
            #"block_13_project_BN", 
            "block_14_add", 
            "block_15_add",
            "block_16_project_BN"
        ]
    ]
    return backbone.input, [c3_output, c4_output, c5_output]


def get_model(input_size = (256,256,3), weights='imagenet'):
    base_inputs, base_outputs = get_backbone_MobileNet(input_shape=input_size, weights=weights)
    c3_output, c4_output, c5_output = base_outputs

    p3_output = Conv2D(256, 1, strides=1, activation='relu', padding="same")(c3_output)
    p4_output = Conv2D(256, 1, strides=1, activation='relu', padding="same")(c4_output)
    p5_output = Conv2D(256, 1, strides=1, activation='relu', padding="same")(c5_output)

    p4_output = Lambda(lambda x: x + p5_output)(p4_output)
    p3_output = Lambda(lambda x: x + p4_output)(p3_output)

    p3_output = Conv2D(256, 3, strides=1, activation='relu', padding="same")(p3_output)
    p4_output = Conv2D(256, 3, strides=1, activation='relu', padding="same")(p4_output)
    p5_output = Conv2D(256, 3, strides=1, activation='relu', padding="same")(p5_output)

    p6_output = Conv2D(256, 3, strides=2, activation='relu', padding="same")(c5_output)
    p7_output = Conv2D(256, 3, strides=2, activation='relu', padding="same")(p6_output)

    p3_output = Flatten()(p3_output)
    p4_output = Flatten()(p4_output)
    p5_output = Flatten()(p5_output)
    p6_output = Flatten()(p6_output)
    p7_output = Flatten()(p7_output)

    m1_output = Concatenate(axis=1)([p3_output,
                                     p4_output,
                                     p5_output,
                                     p6_output,
                                     p7_output])

    #m1_output = Flatten()(m1_output)

    m1_output = Dense(64, activation='relu', kernel_initializer='he_uniform')(m1_output)
    m1_output = Dense(4, activation='sigmoid', kernel_initializer='he_normal')(m1_output)

    model = Model(inputs=base_inputs, outputs=m1_output)

    return model
