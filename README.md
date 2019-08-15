# upper

- 基于seq2seq（TensorFlow）模型 把 小写字母转化成大写字母
- TensorFlow 版本 1.14



## 文件结构

├── Config.py              配置文件

├── Log.py                   日志

├── data                       数据文件夹

├── data_precess.py   数据处理

├── log                          log日志文件夹

├── models                  模型存储文件夹

├── seq2seq_tf.py       seq2seq模型文件

├── test.py                    测试文件

└── train.py                 训练文件



## 数据

采用随机生成小写字母的字符

默认随机生成 15000条长度为5到20的小写随机字符串(a -> z)



### 模型

模型采用seq2seq模型进行训练

seq2seq详细原理可到网络上查找



### 使用方法

训练

```
python train.py 
```

测试

```
python test.py
```





