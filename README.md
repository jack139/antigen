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
