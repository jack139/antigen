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
