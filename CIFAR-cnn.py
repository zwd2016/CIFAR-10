# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:57:17 2018

@author: WenDong Zheng
"""

from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D 
import matplotlib.pyplot as plt

np.random.seed(10)
(x_img_train,y_label_train),(x_img_test,y_label_test) = cifar10.load_data()
print('train data:','images:',x_img_train.shape,'labels:',y_label_train.shape)
print('test data:','images:',x_img_test.shape,'labels:',y_label_test.shape)

x_img_train_normalize = x_img_train.astype('float32')/255.0
x_img_test_normalize = x_img_test.astype('float32')/255.0
x_img_test_normalize1 = x_img_test.astype('float32')/255.0

y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
y_label_test_OneHot1 = np_utils.to_categorical(y_label_test)

#model-adam
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(10,activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x_img_train_normalize,y_label_train_OneHot,validation_split=0.2,epochs=10,batch_size=128,verbose=1)

# plot history loss
plt.ylabel("Loss function value")  
plt.xlabel("The number of epochs")  
plt.title("Loss function-Epoch Curves")
plt.plot(train_history.history['loss'], label='train_adam')
plt.plot(train_history.history['val_loss'], label='val_adam')
plt.legend()
plt.show()

# plot history acc
plt.ylabel("Accuracy value")  
plt.xlabel("The number of epochs")  
plt.title("accuracy-Epoch Curves")
plt.plot(train_history.history['acc'], label='train_adam')
plt.plot(train_history.history['val_acc'], label='val_adam')
plt.legend()
plt.show()

scores = model.evaluate(x_img_test_normalize,y_label_test_OneHot)
print()
print('score_adam: %.6f' % scores[1])
print()

