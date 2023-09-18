#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.engine.sequential import Sequential

RANDOM_SEED = 42


# # Specify each path

# In[3]:


dataset = 'Data2.csv'
model_save_path = 'Skeleton_Input.hdf5'
tflite_save_path = 'Skeleton_Input.tflite'


# # Set number of classes

# In[4]:


NUM_CLASSES = 24


# # Dataset reading

# In[5]:


X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))


# In[6]:


y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.80, random_state=RANDOM_SEED)


# In[8]:


X_train[0].shape


# # Model building

# In[9]:


from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(42, ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(20, activation='tanh'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])


# In[10]:


model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)


# In[11]:


# Model checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=True)
# Callback for early stopping
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)


# In[12]:


# Model compilation
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy', metrics=['acc', 'mse'])


# # Model training

# In[14]:


# model.fit(
#     X_train,
#     y_train,
#     epochs=100,
#     batch_size=128,
#     validation_data=(X_test, y_test),
#     callbacks=[cp_callback, es_callback]
# )

hist = model.fit(X_train, y_train, epochs=350, batch_size=128, validation_data=(X_test,y_test), callbacks=[cp_callback, es_callback])


# In[15]:


import matplotlib.pyplot as plt


loss, acc, mse = model.evaluate(X_test, y_test)
print(f"Loss is {loss},\nAccuracy is {acc * 100},\nMSE is {mse}")


# In[16]:


plt.plot(hist.history['loss'], label = 'loss')
plt.plot(hist.history['val_loss'], label='val loss')
plt.title('Loss vs Val_Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[17]:


plt.figure(figsize=(15,8))
plt.plot(hist.history['acc'], label = 'acc')
plt.plot(hist.history['val_acc'], label='val acc')
plt.title("acc vs Val_acc")
plt.xlabel("Epochs")
plt.ylabel("acc")
plt.legend()
plt.show()


# In[18]:


# Loading the saved model
model = tf.keras.models.load_model(model_save_path)


# In[19]:


# Inference test
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))


# # Confusion matrix

# In[21]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)


# # Convert to model for Tensorflow-Lite

# In[22]:


# Save as a model dedicated to inference
model.save(model_save_path, include_optimizer=False)


# In[23]:


# Transform model (quantization)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)


# # Inference test

# In[24]:


interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()


# In[25]:


# Get I / O tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[26]:


print (input_details)


# In[27]:


print (output_details)


# In[28]:


interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))


# In[29]:


get_ipython().run_cell_magic('time', '', "# Inference implementation\ninterpreter.invoke()\ntflite_results = interpreter.get_tensor(output_details[0]['index'])")


# In[30]:


print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))

