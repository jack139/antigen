# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from data import dataGenerator
from model import get_model
from loss import iou_loss
from metrics import iou_metric

train_dir = '../data/onebox/train'
train_json = '../data/onebox/train_json'
val_dir = '../data/onebox/dev'
val_json = '../data/onebox/dev_json'


model_type = 'vgg16'
freeze = False # 是否冻结 CNN 模型
input_size = (256,256,3)  # 模型输入图片尺寸
batch_size = 128
epochs = 30
learning_rate = 1e-4
train_num = len(os.listdir(train_dir)) # 训练集 数量
val_num = len(os.listdir(val_dir))
train_steps_per_epoch = train_num // batch_size + 1 
val_steps_per_epoch = val_num // batch_size + 1 


# 数据生成器
train_generator = dataGenerator(train_dir, train_json, batch_size=batch_size, target_size=input_size[:2])
val_generator = dataGenerator(val_dir, val_json, batch_size=batch_size, target_size=input_size[:2])


# 生成模型
model = get_model(model_type, input_size=input_size, freeze=freeze)
#model = get_model(model_type, input_size=input_size, freeze=True, weights=None) # for test

model.compile(loss=iou_loss, optimizer=Adam(lr=learning_rate), metrics=[iou_metric])

print(model.summary())

print(f"train data: {train_num}\tdev data: {val_num}")

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")

model_checkpoint = ModelCheckpoint(
	"locate_onebox_%s_b%d_e{epoch:02d}_{val_iou_metric:.5f}.h5"%(model_type, batch_size), 
    monitor='val_iou_metric',verbose=1, save_best_only=True, save_weights_only=True, mode='max'
)

#model.load_weights("./locate_onebox_vgg16_b128_e30_141.h5")

model.fit_generator(train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch,
    callbacks=[model_checkpoint]
)
