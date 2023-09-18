#!/usr/bin/env python
# coding: utf-8

# In[15]:


import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import numpy
import random

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# In[16]:


num_classes = 2
img_rows, img_cols = 28, 28
batch_size = 32

train_data_dir = './Leap_data/train'
validation_data_dir = './Leap_data/test'

# using Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        color_mode = 'grayscale',
        class_mode= 'binary')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        color_mode = 'grayscale',
        class_mode= 'binary')


# In[17]:


model =  Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28,28,1) ))
model.add (MaxPooling2D(pool_size =(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
model.add (MaxPooling2D(pool_size =(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
model.add (MaxPooling2D(pool_size =(2,2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(1, activation = 'sigmoid'))

print(model.summary())

# model = models.alexnet(pretrained=True)
# num_ftrs2 = model.classifier[6].in_features #must use the name same as model_ft -> can try to display
# # Here the size of each output sample is set to 10.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model.classifier[6] = nn.Linear(num_ftrs2, 10)

# # 2. LOSS AND OPTIMIZER
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # 3. move the model to GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)


# In[ ]:


model.compile(loss = 'binary_crossentropy',
             optimizer ='rmsprop',
             metrics =['accuracy'])

nb_train_samples = 101027
nb_validation_samples= 16410

epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)


# In[ ]:


model.save("PicInput_SLR_cnn.h5")


# In[ ]:


from tensorflow.keras.models import load_model

classifier = load_model('PicInput_SLR_cnn.h5')


# In[ ]:


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    frame.cv2.flip(frame, 1)
    
    # define region of interest
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28,28), interpolation = cv2.INTER.AREA)
    
    cv2.imshow('roi scaled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)
    
    roi = roi.reshape(1,28,28,1)
    roi = roi/255
    result = str(classifier.predict_classes(roi, 1, verbose =0 )[0])
    cv2.imshow('frame', copy)
    
    if cv2.waitKey(1) == 13 #13 is the enter key
        break
        
cap.release()
cv2.destroyAllWindows


# In[ ]:




