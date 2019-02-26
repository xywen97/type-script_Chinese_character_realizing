###人工神经网络完成字体的分类鉴定

1、类比于MNIST手写数字集的识别，完成汉字的不同字体的分类识别

2、汉字的字体识别从字体库中获取

3、每个字体可以获取114个字，尽量包含汉字中的笔画的所有变化

4、使用卷积神经网络，对汉字图片进行基本的特征提取

5、标签类似于MNIST手写数字集的处理，为不同的字体的名称（宋体，黑体，等。。。）

6、完成训练分类识别
-------------------------------------------------------------------------------------
运行环境配置：
GPU TITAN xp、python 2.1.15(anaconda)，keras-gpu=2.2.4, tensorflow-gpu=1.8.0
其他依赖包...

下面是参与训练的相关字体库：
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

下面是本项目基本结构图：
writing_word_realizing:
  |
  |--datasets/
  |  |--testlabels/
  |  |  |--testlabels.txt
  |  |--testsets/
  |  |  |--001
  |  |  |  |--000.jpg
  |  |  |  |--001.jpg
  |  |  |  |--xxx.jpg
  |  |  |--002
  |  |  |  |--000.jpg
  |  |  |  |--001.jpg
  |  |  |  |--xxx.jpg
  |  |  |--...
  |  |--trainlabels/
  |  |  |--trainlabels.txt
  |  |--trainsets/
  |     |--001
  |     |  |--000.jpg
  |     |  |--001.jpg
  |     |  |--xxx.jpg
  |     |--002
  |     |  |--000.jpg
  |     |  |--001.jpg
  |     |  |--xxx.jpg
  |     |--...
  |--traind_model/
  |  |--model.h5
  |--ttf_sets/
  |  |--xxx.ttf/TTF
  |  |--xxx.ttc/TTC
  |--demo.py
  |--mnist.py
  |--README.md
  |--reshape_img.py
  |--str2img.py

预处理：
数据集生成：
  文字转图片：
  python str2img.py 
  图片大小设定（28*28）：
  python reshape_img.py
  
运行：
python demo.py

测试：
 。。。。。。
