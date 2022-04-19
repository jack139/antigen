# coding=utf-8

import numpy as np
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, Flatten, Concatenate, Dense
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
    return Model(
        inputs=backbone.input, outputs=[c3_output, c4_output, c5_output]
    )

class customFeaturePyramid2(Model):
    """Builds the Feature Pyramid with the feature maps from the backbone.
    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50.
    """
    def __init__(self, backbone=None, class_number=4, **kwargs):
        super(customFeaturePyramid2, self).__init__(name="customFeaturePyramid2", **kwargs)
        self.backbone = backbone
        self.class_number = class_number
        self.conv_c3_1x1 = Conv2D(256, (1, 1), padding="same", input_shape=(32, 32, 512))
        self.conv_c4_1x1 = Conv2D(256, (1, 1), padding="same", input_shape=(16, 16, 1024))
        self.conv_c5_1x1 = Conv2D(256, (1, 1), padding="same", input_shape=(8, 8, 2048))
        self.conv_c3_3x3 = Conv2D(256, (3, 1), padding="same")
        self.conv_c4_3x3 = Conv2D(256, (3, 1), padding="same")
        self.conv_c5_3x3 = Conv2D(256, (3, 1), padding="same")
        self.conv_c6_3x3 = Conv2D(256, (3, 2), padding="same")
        self.conv_c7_3x3 = Conv2D(256, (3, 2), padding="same")
        self.upsample_2x = UpSampling2D(2)
        self.dense_d1 = Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.dense_d2 = Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.dense_d3 = Dense(64, activation='relu', kernel_initializer='he_uniform')
        self.dense_d4 = Dense(self.class_number, activation='sigmoid', kernel_initializer='he_normal')

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
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
        m1_output = Flatten()(m1_output)
        m1_output = self.dense_d1(m1_output)
        m1_output = self.dense_d2(m1_output)
        m1_output = self.dense_d3(m1_output)
        m1_output = self.dense_d4(m1_output)
        return m1_output

def get_model(model_type='resnet', input_size = (224,224,3), freeze=False, weights='imagenet'):
    resnet50Backbone = get_backbone_ResNet50(input_shape=input_size, weights=weights)
    model = customFeaturePyramid2(resnet50Backbone, 4)
    return model
