# 抗原检测结果识别



## 生成定位训练数据

```
cd datagen
python3 datagen.py
```



## 定位模型训练

```
cd locate
python3 cnn_train.py
```



## 生成识别训练数据

```
cd locate
python3 predict <datagen生成图片路径>
```



## 识别模型训练

```
cd detpos
python3 train_net.py
```



## 识别

```
python3 predict_flow.py <图片路径>
```



## 编译 sm3
```
cd api/utils/libsm3
gcc -fPIC -shared -o libsm3.so sm3.c
```



## 训练过程
1. 生成locate训练集、验证集（datagen/datagen.py）
2. 加入plus/imags数据
3. 训练locate模型（locate/cnn_train.py）
4. 生成detpos训练集、验证集（datagen/datagen.py）
5. 加入plus/imgs数据
6. 使用locate模型跑4/5生成的数据，生成实际detpos训练数据（locate/predict.py）
7. 检查各分类结果，将iou较小的数据归入nul
8. 加入plus/detpos_imgs数据
9. 训练detpos模型（detpos/train_net.py）
