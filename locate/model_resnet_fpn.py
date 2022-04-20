# coding=utf-8

from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, Flatten, Concatenate, Dense, Lambda, Input
import tensorflow as tf


def get_backbone_ResNet50(input_shape, weights):
    """Builds ResNet50 with pre-trained imagenet weights"""
    tf.reset_default_graph() # 防止 layer name 变化
    backbone = ResNet50(include_top=False, input_shape=input_shape, weights=weights)
    #print('\n'.join([l.name for l in backbone.layers]))
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["activation_22", "activation_40", "activation_49"] # names in keras 2.3.1
    ]
    # ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"] # names in tf.keras
    return backbone.input, [c3_output, c4_output, c5_output]


def get_model(input_size = (224,224,3), weights='imagenet'):
    base_inputs, base_outputs = get_backbone_ResNet50(input_shape=input_size, weights=weights)
    c3_output, c4_output, c5_output = base_outputs

    p3_output = Conv2D(256, (1, 1), activation='relu', padding="same")(c3_output)
    p4_output = Conv2D(256, (1, 1), activation='relu', padding="same")(c4_output)
    p5_output = Conv2D(256, (1, 1), activation='relu', padding="same")(c5_output)

    p4_output = Lambda(lambda x: x + UpSampling2D((2,2))(p5_output))(p4_output)
    p3_output = Lambda(lambda x: x + UpSampling2D((2,2))(p4_output))(p3_output)

    p3_output = Conv2D(256, (3, 3), activation='relu', padding="same")(p3_output)
    p4_output = Conv2D(256, (3, 3), activation='relu', padding="same")(p4_output)
    p5_output = Conv2D(256, (3, 3), activation='relu', padding="same")(p5_output)

    p6_output = Conv2D(256, (3, 3), activation='relu', padding="same")(c5_output)
    p7_output = Conv2D(256, (3, 3), activation='relu', padding="same")(p6_output)

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

    m1_output = Dense(64, activation='relu', kernel_initializer='he_uniform')(m1_output)
    m1_output = Dense(4, activation='sigmoid', kernel_initializer='he_normal')(m1_output)

    model = Model(inputs=base_inputs, outputs=m1_output)

    return model
