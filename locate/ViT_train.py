import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback
from model_ViT import VisionTransformer
from data import dataGenerator


train_dir = '../data/onebox/train'
train_json = '../data/onebox/train_json'
val_dir = '../data/onebox/dev'
val_json = '../data/onebox/dev_json'


input_size = (256,256)  # 模型输入图片尺寸
batch_size = 32
learning_rate = 8e-4
epochs = 30
train_num = len(os.listdir(train_dir)) # 训练集 数量
val_num = len(os.listdir(val_dir))
train_steps_per_epoch = train_num // batch_size + 1 
val_steps_per_epoch = val_num // batch_size + 1 

'''
Model  Layers Hidden_size_D MLP_size    Heads   Params
Base    12      768         3072        12      86M
Large   24      1024        4096        16      307M
Huge    32      1280        5120        16      632M
'''
layers = 12
d_model = 768   # Hidden_size_D
n_head = 12
mlp_dim = 3072   # MLP_size
patch_size = 16
dropout = 0.1
classes_num = 4

# 数据生成器
train_generator = dataGenerator(train_dir, train_json, batch_size=batch_size, target_size=input_size[:2])
val_generator = dataGenerator(val_dir, val_json, batch_size=batch_size, target_size=input_size[:2])


# 生成模型
vit = VisionTransformer(
    input_size[0], 
    classes_num, 
    d_model=d_model, 
    d_inner_hid=mlp_dim,
    n_head=n_head, 
    layers=layers, 
    dropout=dropout, 
    patch_size=patch_size
)

vit.compile(Adam(learning_rate, 0.9, 0.98, epsilon=1e-9))

vit.model.summary()

print(f"train data: {train_num}\tdev data: {val_num}")

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")

class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = d_model**-0.5
        self.warm = warmup**-1.5
        self.step_num = 0
    def on_batch_begin(self, batch, logs = None):
        self.step_num += 1
        lr = self.basic * min(self.step_num**-0.5, self.step_num*self.warm)
        K.set_value(self.model.optimizer.lr, lr)
    def on_epoch_begin(self, epoch, logs = None):
        print('lr=', K.get_value(self.model.optimizer.lr))
lr_scheduler = LRSchedulerPerStep(d_model, 4000) 

model_checkpoint = ModelCheckpoint(
    "locate_onebox_ViT_b%d_e{epoch:02d}_{val_iou_metric:.5f}.h5"%(batch_size), 
    monitor='val_iou_metric',verbose=1, save_best_only=True, save_weights_only=True, mode='max'
)

vit.model.fit_generator(train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch,
    callbacks=[model_checkpoint, lr_scheduler]
)
