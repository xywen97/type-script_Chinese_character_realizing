#-*- coding = utf-8 -*-

import keras
import tensorflow as tf
import os
import matplotlib
import numpy as np

from scipy.misc import imread,imshow
from keras.models import load_model
from keras import backend as K

# limit gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess_config = tf.ConfigProto() 
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=sess_config) 

# reload the trained model
model = load_model('trained_model/model.h5')

img_rows, img_cols = 28, 28
test = []
testsets_path = 'datasets/testimages/'

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# to trans the rgb images to gray images
def gci(sets, filepath):
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)   
    print(fi_d)         
    if os.path.isdir(fi_d):
      gci(sets, fi_d)                  
    else:
      img = imread(fi_d)
      img = rgb2gray(img)
      sets.append(img)

# load datasets
gci(test,testsets_path)
# from list to numpy array
test = np.array(test)

# reshape the images
if K.image_data_format() == 'channels_first':
    test = test.reshape(test.shape[0], 1, img_rows, img_cols)
    # x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    test = test.reshape(test.shape[0], img_rows, img_cols, 1)
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

test /= 255
print('test shape:', test.shape)
print(test.shape[0], 'test samples')

# for item in test:
#     a = model.predict(item, batch_size=1, verbose=0)
#     print(a)
a = model.predict(test, batch_size=1, verbose=0)
for i in a:
    print(np.argmax(i))
# print(a)