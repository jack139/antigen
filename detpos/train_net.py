# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, InceptionResNetV2, VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import Dense, GlobalAveragePooling2D, Dropout


input_size = (128,128,3) # 对 crop 数据 128就够了
batch_size = 16
steps_per_epoch = 20
epochs = 10
train_dir = '../data/crop_train'
test_dir = '../data/crop_dev'


# 数据生成器
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x.astype(np.float32),
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    class_mode="categorical",
    target_size=input_size[:2],
    batch_size=batch_size,
)

print(train_generator.class_indices)

test_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x.astype(np.float32),
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    class_mode="categorical",
    target_size=input_size[:2],
    batch_size=batch_size,
)

print(test_generator.class_indices)

# create the base pre-trained model
base_model = VGG16(weights='imagenet', input_shape=input_size, include_top=False)
#base_model = ResNet50(weights='imagenet', input_shape=input_size, include_top=False)
#base_model = InceptionResNetV2(weights='imagenet', input_shape=input_size, include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
#x = Dropout(0.2)(x)
#x = Dense(32, activation='relu')(x)
# and a logistic layer 
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr = 1e-4), loss='categorical_crossentropy', metrics = ['categorical_accuracy'])

model.summary()

# train the model on the new data for a few epochs
model.fit(train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=5)

model.save('batch_%d_epochs_%d_steps_%d_0.h5'%(batch_size, epochs, steps_per_epoch))

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

print(model.layers)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 100 layers and unfreeze the rest:
for layer in model.layers[:100]:
   layer.trainable = False
for layer in model.layers[100:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['categorical_accuracy'])

#model.summary()

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=5)

model.save('batch_%d_epochs_%d_steps_%d_1.h5'%(batch_size, epochs, steps_per_epoch))