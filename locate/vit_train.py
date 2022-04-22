import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from model_vit import VisionTransformer
from data import dataGenerator
from loss import iou_loss
from metrics import iou_metric


train_dir = '../data/onebox/train'
train_json = '../data/onebox/json'
val_dir = '../data/onebox/dev'
val_json = '../data/onebox/json'

input_size = (256,256,3)  # 模型输入图片尺寸
batch_size = 4
learning_rate = 1e-5
epochs = 30
train_num = len(os.listdir(train_dir)) # 训练集 数量
val_num = len(os.listdir(val_dir))
train_steps_per_epoch = train_num // batch_size + 1 
val_steps_per_epoch = val_num // batch_size + 1 


# 数据生成器
train_generator = dataGenerator(train_dir, train_json, batch_size=batch_size, target_size=input_size[:2])
val_generator = dataGenerator(val_dir, val_json, batch_size=batch_size, target_size=input_size[:2])

# ViT 模型
model = VisionTransformer(
    image_size=input_size[0],
    patch_size=4,
    num_layers=8,
    num_classes=4,
    d_model=64,
    num_heads=4,
    mlp_dim=128,
    channels=3,
    dropout=0.1,
)
model.compile(loss=iou_loss, optimizer=Adam(lr=learning_rate), metrics=[iou_metric])

print(f"train data: {train_num}\tdev data: {val_num}")

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")

model_checkpoint = ModelCheckpoint(
    "locate_onebox_ViT_b%d_e{epoch:02d}_{val_iou_metric:.5f}.h5"%(batch_size), 
    monitor='val_iou_metric',verbose=1, save_best_only=True, save_weights_only=True, mode='max'
)
#reduce_lr = ReduceLROnPlateau(
#    monitor='loss', factor=0.1, patience=3, verbose=0, mode='auto',
#    min_delta=0.0001, cooldown=0, min_lr=0)    

# 训练一次，进行编译，否则 fit_generator 会报模型编译错误
x, y = next(train_generator)
model.fit(np.array([x[0]]), np.array([y[0]]), epochs=1)

#model.load_weights("./locate_onebox_resnet-fpn_b128_e15_0.84804.h5")

model.fit_generator(train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch,
    callbacks=[model_checkpoint]
)
