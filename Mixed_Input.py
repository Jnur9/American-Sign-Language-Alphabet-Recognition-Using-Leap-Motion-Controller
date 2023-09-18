#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Get names of Pokemon pictures

import os

names = os.listdir('leap_data/Images')

names[:24]


# In[2]:


# Get Pandas DataFrame with Name column and path to the image

import pandas as pd

letters_df = pd.DataFrame({'Letter':[x.split('.')[0] for x in names], 
                         'Path':['leap_data/Images/' + x for x in names]})

letters_df


# In[3]:


# Read in bigger Pokemon.csv

skeleton_df2 = pd.read_csv('Data.csv')

skeleton_df2.head()


# In[4]:


# Make name column lower-case to match with other table

skeleton_df2['Letter'] = skeleton_df2['Letter'].apply(lambda s: s.upper())

skeleton_df2


# In[5]:


# Join the two pokemon dataframes on the name column

merged_df = pd.merge(left=letters_df, right=skeleton_df2, on='Letter', how='inner')

merged_df


# In[7]:


#In may case I dont need to drop anything
# Drop unneccessary columns

merged_df = merged_df.drop(['#', 'Type 1', 'Type 2', 'Total',
                              'Generation', 'Legendary'], axis=1)

merged_df.head()


# In[6]:


len(merged_df)


# In[7]:


# Store Data in Compressed NumPy array files (.NPZs)

import numpy as np
import cv2

npz_paths = []

for i, row in merged_df.iterrows():
  picture_path = row['Path']

  npz_path = picture_path.split('.')[0] + '.npz'
  npz_paths.append(npz_path)

  pic_bgr_arr = cv2.imread(picture_path)
  pic_rgb_arr = cv2.cvtColor(pic_bgr_arr, cv2.COLOR_BGR2RGB)

  J0, J1         = row['J0'], row['J1']
  J2, J3, J4 = row['J2'], row['J3'], row['J4']
  J5, J6, J7, J8, J9, J10 = row['J5'], row['J6'], row['J7'], row['J8'], row['J9'], row['J10']
  J11, J12, J13, J14, J15, J16 = row['J11'], row['J12'], row['J13'], row['J14'], row['J15'], row['J16']
  J17, J18, J19, J20, J21, J22 = row['J17'], row['J18'], row['J19'], row['J20'], row['J21'], row['J22']
  J23, J24, J25, J26, J27, J28 = row['J23'], row['J24'], row['J25'], row['J26'], row['J27'], row['J28']
  J29, J30, J31, J32, J33, J34 = row['J29'], row['J30'], row['J31'], row['J32'], row['J33'], row['J34']
  J35, J36, J37, J38, J39, J40, J41 = row['J35'], row['J36'], row['J37'], row['J38'], row['J39'], row['J40'], row['J41']

  stats = np.array([J0, J1, J2, J3, J4, J5, J6, J7, J8, J9, J10, J11, J12, J13, J14, J15, J16, J17, J18, J19,
                   J20, J21, J22, J23, J24, J25, J26, J27, J28, J29, J30, J31, J32, J33, J34, J35, J36, J37,
                   J38, J39, J40, J41])

  hp = row['id']
  np.savez_compressed(npz_path, pic=pic_rgb_arr, stats=stats, hp=hp)

merged_df['NPZ_Path'] = pd.Series(npz_paths)

merged_df.head()


# In[8]:


# Get DataFrame of Stats Only

stats_df = merged_df[['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 
                     'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19',
                     'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26', 'J27', 'J28', 'J29',
                     'J30', 'J31', 'J32', 'J33', 'J34', 'J35', 'J36', 'J37', 'J38', 'J39', 'J40', 'J41']]

stats_df


# In[9]:


# Calculate the mean of each column

means = [stats_df[col].mean() for col in stats_df]

means


# In[10]:


# Calculate the Std. Deviation of each column

std_devs = [stats_df[col].std()+0.000001 for col in stats_df] 

std_devs


# In[11]:


# Create TensorFlow preprocessing function for stats stream

import tensorflow as tf

def stat_scaler(tensor):
  return (tensor - means) / std_devs

stat_scaler(tf.constant([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], dtype=tf.float32))


# In[12]:


# Load example pic and stats from Abomasnow NPZ

B_npz = np.load('Leap_data/Images/B.npz')

B_npz['pic'].shape, B_npz['stats'].shape


# In[115]:


# Display the example image

import matplotlib.pyplot as plt

plt.imshow(B_npz['pic'])


# In[14]:


# Drop stats columns (other than health), and original image, since we have that stored in the image

merged_df.drop(['Path','J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 
                     'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19',
                     'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26', 'J27', 'J28', 'J29',
                     'J30', 'J31', 'J32', 'J33', 'J34', 'J35', 'J36', 'J37', 'J38', 'J39', 'J40', 'J41'], inplace=True, axis=1)

merged_df


# In[109]:


# Shuffle the DataFrame, and split into train, validation and test sets

shuffled_df = merged_df.sample(frac=1)

train_df, val_df, test_df = shuffled_df[:2500], shuffled_df[2500:3500], shuffled_df[3500:]


# In[16]:


# Display train_df

train_df


# In[144]:


# Create function to get X_pic, X_stats, and y from a DataFrame

def get_X_y(df):

  X_pic, X_stats = [], []
  y = []

  for name in df['NPZ_Path']:
    loaded_npz = np.load(name)

    pic = loaded_npz['pic']
    X_pic.append(pic)

    stats = loaded_npz['stats']
    X_stats.append(stats)
    
    y.append(loaded_npz['hp'])

  X_pic, X_stats = np.array(X_pic), np.array(X_stats)
  y = np.array(y)

  return (X_pic, X_stats), y

(X_train_pic, X_train_stats), y_train = get_X_y(train_df)


# Get the training data

(X_train_pic.shape, X_train_stats.shape), y_train.shape


# In[18]:


# Get the val data

(X_val_pic, X_val_stats), y_val = get_X_y(val_df)

(X_val_pic.shape, X_val_stats.shape), y_val.shape


# In[19]:


# Get the test data

(X_test_pic, X_test_stats), y_test = get_X_y(test_df)

(X_test_pic.shape, X_test_stats.shape), y_test.shape


# In[57]:


# Define the Model

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Define the Picture (CNN) Stream

input_pic = layers.Input(shape=(124, 124, 3))
x         = layers.Lambda(preprocess_input)(input_pic)
x         = MobileNetV2(input_shape=((124, 124, 3)), include_top=False)(x)
x         = layers.GlobalAveragePooling2D()(x)
x         = layers.Dense(10, activation='relu')(x)
x         = Model(inputs=input_pic, outputs=x)


# Define the Stats (Feed-Forward) Stream

input_stats = layers.Input(shape=(42,))
y = layers.Lambda(stat_scaler)(input_stats)
y = layers.Dense(64, activation="relu")(y)
y = layers.Dense(10, activation="relu")(y)
y = Model(inputs=input_stats, outputs=y)


# Concatenate the two streams together
combined = layers.concatenate([x.output, y.output])

# Define joined Feed-Forward Layer
z = layers.Dense(4, activation="relu")(combined)

# Define output node of 1 linear neuron (regression task)
z = layers.Dense(1, activation="linear")(z)


# Define the final model
model = Model(inputs=[x.input, y.input], outputs=z)


# In[58]:


# Observe a (confusing) summary of the model

model.summary()


# In[59]:


# Compile the model with Adam optimizer and mean-squared-error loss function

from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mean_absolute_error'])


# In[60]:


# Create a model saving callback and train for 10 epochs (connect to GPU runtime!!)

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


cp = ModelCheckpoint('model/',  save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
model.fit(x=[X_train_pic, X_train_stats], y=y_train, validation_data=([X_val_pic, X_val_stats], y_val), epochs=10, callbacks=[cp, es])


# In[61]:


# Load the saved model

from tensorflow.keras.models import load_model

loaded_model = load_model('model/')


# In[62]:


# Use the loaded model to obtain predictions on the test set

test_predictions = loaded_model.predict([X_test_pic, X_test_stats]).flatten()

test_predictions.shape


# In[63]:


# Convert predictions to a Pandas Series and view histogram
# Notice the model overfits!

test_preds_series = pd.Series(test_predictions)

test_preds_series.hist()


# In[64]:


test_df.head()


# In[65]:


# Reset index column, and add predictions to the test_df

test_df.reset_index(drop=True, inplace=True)

test_df['Predicted ID'] = test_preds_series

test_df


# In[66]:


# Compare results to linear regression on stats-only from scikit-learn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

linear_model = LinearRegression().fit(X_train_stats, y_train)

mean_absolute_error(y_test, linear_model.predict(X_test_stats))


# In[68]:


mean_absolute_error(y_test, test_preds_series)


# In[67]:


# Add predictions from linear model to the test dataframe

test_df['Linear Predicted id'] = pd.Series(linear_model.predict(X_test_stats))

test_df


# In[69]:


# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save('Mixed_Input_SLR.h5')
print("Done !")


# In[125]:


tflite_save_path = 'Mixed_Input.tflite'

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)


# In[126]:


interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()


# In[128]:


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[129]:


print (input_details)


# In[130]:


print (output_details)

