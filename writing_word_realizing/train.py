#-*- coding: utf-8 -*- 

from __future__ import print_function
import keras
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import matplotlib
from scipy.misc import imread,imshow
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess_config = tf.ConfigProto() 
#sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=sess_config) 

batch_size = 4
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = []
y_train = []
x_test = []
y_test = []

trainsets_path = 'datasets/trainsets/'
testsets_path = 'datasets/testsets/'
trainlabels_path = 'datasets/trainlabels/'
testlabels_path = 'datasets/testlabels/'
'''

'''

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# to trans the rgb images to gray images
def gci(sets, filepath):
  files = os.listdir(filepath)
  files.sort()
  for fi in files:
    fi_d = os.path.join(filepath,fi)     
    print(fi_d)       
    if os.path.isdir(fi_d):
      gci(sets, fi_d)                  
    else:
      img = imread(fi_d)
      img = rgb2gray(img)
      sets.append(img)
      #print(len(sets))
      #print(imread(fi_d))

gci(x_train, trainsets_path)
# print(x_train[0])
gci(x_test, testsets_path)

x_train = np.array(x_train)
x_test = np.array(x_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#x_train = x_train.astype('float32')
#print(x_train[0][0])
#x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = []
y_test = []
# convert class vectors to binary class matrices
with open('datasets/trainlabels/trainlabels.txt') as f:
  for line in f:
    line = line.strip('\r\n')
    line_1 = map(int, line.split(','))
    y_train.append(line_1)

#y_train = keras.utils.to_categorical(y_train, num_classes)
#print(y_train[0:2])
#y_test = keras.utils.to_categorical(y_test, num_classes)'
with open('datasets/testlabels/testlabels.txt') as f:
  for line in f:
    line = line.strip('\r\n')
    line_1 = map(int, line.split(','))
    y_test.append(line_1)

y_train = np.array(y_train)
y_test = np.array(y_test)
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

'''
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same',input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3),activation='relu', padding='same', strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3),activation='relu', padding='same', strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save('trained_model/model.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])