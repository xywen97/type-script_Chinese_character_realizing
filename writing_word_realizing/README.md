## 人工神经网络完成字体的分类鉴定  

1、类比于MNIST手写数字集的识别，完成汉字的不同字体的分类识别  

2、汉字的字体识别从字体库中获取  
 
3、每个字体可以获取114个字(本工程中只选取了这么多)，尽量包含汉字中的笔画的所有变化  

4、使用卷积神经网络，对汉字图片进行基本的特征提取  

5、标签类似于MNIST手写数字集的处理，为不同的字体的名称（宋体，黑体，等。。。）  

6、完成训练

### 运行环境配置：  
GPU TITAN xp、python 2.1.15(anaconda)，keras-gpu=2.2.4, tensorflow-gpu=1.8.0  
其他依赖包...  

### 下面是参与训练的相关字体库：  
ttf_sets:  
DENG.TTF  
FZSTK.TTF  
FZYTK.TTF  
MSYH.TTC  
MSYHBD.TTC  
MSYHL.TTC   
SIMLI.TTF  
SIMYOU.TTF  
STHUPO.TTF  
STSONG.TTF  

### 下面是本项目基本结构图：  
![image](https://github.com/Toneywen/writing_character_realizing/blob/master/writing_word_realizing/configs/%E7%BB%93%E6%9E%84%E5%9B%BE.jpg)
  
### 预处理：  
数据集生成：  
  文字转图片：  
  python str2img.py   
    
  图片大小设定（28*28）：  
  python reshape_img.py  
    
  
### 运行：  
python demo.py  
  
### 测试：  
python test.py
测试数据集包含在 datasets/testimages下面
测试数据集生成：使用str2img.py生成

### 运行结果：
#### 训练阶段：
1140/1140 [==============================] - 3s 3ms/step - loss: 1.8116 - acc: 0.3211 - val_loss: 0.7419 - val_acc: 0.7167  
Epoch 2/10  
1140/1140 [==============================] - 2s 2ms/step - loss: 0.6283 - acc: 0.7746 - val_loss: 0.2111 - val_acc: 0.9500  
Epoch 3/10  
1140/1140 [==============================] - 2s 2ms/step - loss: 0.2964 - acc: 0.8921 - val_loss: 0.1274 - val_acc: 0.9333  
Epoch 4/10  
1140/1140 [==============================] - 4s 4ms/step - loss: 0.1810 - acc: 0.9342 - val_loss: 0.1197 - val_acc: 0.9500  
Epoch 5/10  
1140/1140 [==============================] - 5s 4ms/step - loss: 0.1145 - acc: 0.9588 - val_loss: 0.0270 - val_acc: 1.0000  
Epoch 6/10  
1140/1140 [==============================] - 5s 5ms/step - loss: 0.0863 - acc: 0.9754 - val_loss: 0.1257 - val_acc: 0.9833  
Epoch 7/10  
1140/1140 [==============================] - 5s 4ms/step - loss: 0.0724 - acc: 0.9763 - val_loss: 0.1124 - val_acc: 0.9500  
Epoch 8/10  
1140/1140 [==============================] - 5s 4ms/step - loss: 0.0488 - acc: 0.9842 - val_loss: 0.0128 - val_acc: 1.0000  
Epoch 9/10  
1140/1140 [==============================] - 5s 4ms/step - loss: 0.0434 - acc: 0.9860 - val_loss: 0.0060 - val_acc: 1.0000  
Epoch 10/10  
1140/1140 [==============================] - 5s 4ms/step - loss: 0.0364 - acc: 0.9877 - val_loss: 0.0928 - val_acc: 0.9833  
Test loss: 0.0927583186577  
Test accuracy: 0.983333333333  
训练结果看起来有些过拟合.....

#### 测试阶段
datasets/testimages/004.jpg  
datasets/testimages/009.jpg  
datasets/testimages/001.jpg  
datasets/testimages/006.jpg  
datasets/testimages/005.jpg  
datasets/testimages/003.jpg  
datasets/testimages/000.jpg  
datasets/testimages/002.jpg  
datasets/testimages/008.jpg  
datasets/testimages/007.jpg  
('test shape:', (10, 28, 28, 1))  
(10, 'test samples')  
4  
9  
1  
6  
5  
3  
0  
2  
8  
7  
经过softmax分类过后，与投喂的测试数据集类别吻合，泛化能力还行...
