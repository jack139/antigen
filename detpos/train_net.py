# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, MobileNetV2, VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint


freeze = False # 是否冻结 CNN 模型
input_size = (128,128,3) # 对 crop 数据 128就够了
batch_size = 512
learning_rate = 1e-4
train_num = 9000
dev_num = 1000
train_steps_per_epoch = train_num // batch_size + 1 
dev_steps_per_epoch = dev_num // batch_size + 1 
epochs = 10
train_dir = '../data/crop_train'
test_dir = '../data/crop_dev'


# 数据生成器
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x.astype(np.float32),
    rotation_range=3,
    width_shift_range=5,
    height_shift_range=5,
    zoom_range=0.1
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    class_mode="categorical",
    target_size=input_size[:2],
    batch_size=batch_size,
)


test_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x.astype(np.float32),
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    class_mode="categorical",
    target_size=input_size[:2],
    batch_size=batch_size,
)


# create the base pre-trained model
base_model = VGG16(weights='imagenet', input_shape=input_size, include_top=False)
#base_model = MobileNetV2(weights='imagenet', input_shape=input_size, include_top=False)
#base_model = ResNet50(weights='imagenet', input_shape=input_size, include_top=False)


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# fully-connected layer
x = Dense(512, activation='relu')(x)
#x = Dropout(0.2)(x)
#x = Dense(32, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
if freeze:
    for layer in base_model.layers:
        layer.trainable = False

model.compile(optimizer=Adam(lr = learning_rate), 
    loss='categorical_crossentropy', metrics = ['categorical_accuracy'])

model.summary()

print(train_generator.class_indices)

ckpt_filepath = "detpos_onebox_%s_b%d_e%d_%d.h5"%('vgg16',batch_size,epochs,train_steps_per_epoch)

model_checkpoint = ModelCheckpoint(ckpt_filepath, 
    monitor='val_categorical_accuracy',verbose=1, save_best_only=True, 
    save_weights_only=True, mode='max')

# train the model on the new data for a few epochs
model.fit(train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=dev_steps_per_epoch,
        callbacks=[model_checkpoint]
)

