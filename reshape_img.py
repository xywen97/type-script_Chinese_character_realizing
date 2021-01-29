#-*- coding:utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sess_config = tf.ConfigProto() 
#sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=sess_config) 

path = 'datasets'

def gci(filepath):
#遍历filepath下所有文件，包括子目录
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)            
    if os.path.isdir(fi_d):
      gci(fi_d)                  
    else:
      print os.path.join(filepath,fi_d)
      image_raw_data = tf.gfile.FastGFile(fi_d, 'rb').read()
      with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        print("file_name: "+fi)
  
        resized = tf.image.resize_images(img_data, [28,28],method=0)  
        #第一个参数为原始图像，第二个参数为图像大小，第三个参数给出了指定的算法
        resized = np.asarray(resized.eval(),dtype='uint8')
        #plt.imshow(resized)
        print("picture tramsformed!")
        #plt.show()
       
        encoded_image = tf.image.encode_jpeg(resized)
        print("cnm")   # 中间老报错，贼气
        with tf.gfile.GFile(fi_d, 'wb') as f:
            f.write(encoded_image.eval())
      
if __name__ == '__main__':
  gci(path)