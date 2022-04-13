# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from data import dataGenerator
from model import get_model
from loss import IoULoss
from metrics import IoU, IoU2

train_dir = '../data/train'
train_json = '../data/json'
val_dir = '../data/dev'
val_json = '../data/json'


input_size = (256,256,3)
train_num = len(os.listdir(train_dir)) # 训练集 数量
val_num = len(os.listdir(val_dir))
batch_size = 16
train_steps_per_epoch = train_num // batch_size + 1 
val_steps_per_epoch = val_num // batch_size + 1 
epochs = 10


# 数据生成器
train_generator = dataGenerator(train_dir, train_json, batch_size=batch_size, target_size=input_size[:2])
val_generator = dataGenerator(val_dir, val_json, batch_size=batch_size, target_size=input_size[:2])


# 生成模型
model_type = 'vgg16'
#model = get_model(model_type, input_size=input_size, freeze=True)
model = get_model(model_type, input_size=input_size, freeze=True, weights=None) # for test

opt = Adam(lr=1e-4)
#model.compile(loss="mse", optimizer=opt, metrics=[IoU, IoU2])
model.compile(loss=IoULoss, optimizer=opt, metrics=[IoU, IoU2])

print(model.summary())

print(f"train data: {train_num}\tdev data: {val_num}")

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")

ckpt_filepath = "locate_%s_b%d_e%d_%d.h5"%(model_type,batch_size,epochs,train_steps_per_epoch)

model_checkpoint = ModelCheckpoint(ckpt_filepath, 
    monitor='val_IoU',verbose=1, save_best_only=True, save_weights_only=True, mode='max')

model.fit_generator(train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch,
    callbacks=[model_checkpoint]
)

# 解冻base model的参数后再训练

# 数据生成器
train_generator = dataGenerator(train_dir, train_json, batch_size=batch_size, target_size=input_size[:2])
val_generator = dataGenerator(val_dir, val_json, batch_size=batch_size, target_size=input_size[:2])

# 解冻的模型
model = get_model(model_type, input_size=input_size, freeze=False, weights=None)
model.load_weights(ckpt_filepath)

model.compile(loss=IoULoss, optimizer=opt, metrics=[IoU, IoU2])

print(model.summary())

model.fit_generator(train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch,
    callbacks=[model_checkpoint]
)
