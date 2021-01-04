import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pylab as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import PIL
from PIL import ImageFilter
import cv2
import itertools
import random
import keras
import imutils
from imutils import paths
import os
from keras import optimizers
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import callbacks
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose
from keras import backend as K

ree = 0
getter1 = pandas.read_csv('test-nfl-spread.csv')
getter2 = getter1.values
train = np.zeros([1, getter2.shape[1]])
for i in range(len(getter2)):
    try:
        if not getter2[i, 3] == '-' and not math.isnan(float(getter2[i, 3])):
            train = np.vstack((train, getter2[i, :]))
    except:
        pass

X_trn_raw, y_trn = train[1:,3:21], train[1:,23]
spread = train[1:,30]
cover = train[1:,31]
names = train[1:,1]
y_trn = np.array([float(item) for item in y_trn])
y_trn = y_trn.astype(np.float32)
spread = np.array([float(item) for item in spread])
spread = spread.astype(np.float32)
cover = np.array([int(item) for item in cover])
cover = cover.astype(np.int)
X_trn_raw = np.array([[float(elm) for elm in row] for row in X_trn_raw])
X_trn_raw = X_trn_raw.astype(np.float32)
for i in range(len(X_trn_raw)):
    X_trn_raw[i, 1] = X_trn_raw[i, 0] - X_trn_raw[i, 1]
    X_trn_raw[i, 10] = X_trn_raw[i, 9] - X_trn_raw[i, 10]

pandas.DataFrame(X_trn_raw).to_csv("raw.csv")

for i in range(len(cover)):
    if cover[i] == -1:
        cover[i] = 0

X_trn_raw = (X_trn_raw * 1000).astype(int)
X_trn_raw = (X_trn_raw.astype(np.float32)/1000.0)
 
X_tst_linear, spread_tst, cover_tst, names_tst = X_trn_raw[544:,:], spread[544:], cover[544:], names[544:]
X_trn_linear, spread_trn, cover_trn, names_trn = X_trn_raw[94:480,:], spread[94:480], cover[94:480], names[94:480]

X_tst_linear = np.concatenate((X_tst_linear, np.array([spread_tst]).T), axis=1)
X_trn_linear = np.concatenate((X_trn_linear, np.array([spread_trn]).T), axis=1)

X_tst_linear_large = np.zeros(X_tst_linear.shape)
X_tst_linear_small = np.zeros(X_tst_linear.shape)
spread_tst_large = np.zeros(spread_tst.shape)
spread_tst_small = np.zeros(spread_tst.shape)
cover_tst_large = np.zeros(cover_tst.shape)
cover_tst_small = np.zeros(cover_tst.shape)
names_tst_large = np.ndarray(names_tst.shape, str)
names_tst_small = np.ndarray(names_tst.shape, str)
X_trn_linear_large = np.zeros(X_trn_linear.shape)
X_trn_linear_small = np.zeros(X_trn_linear.shape)
spread_trn_large = np.zeros(spread_trn.shape)
spread_trn_small = np.zeros(spread_trn.shape)
cover_trn_large = np.zeros(cover_trn.shape)
cover_trn_small = np.zeros(cover_trn.shape)
names_trn_large = np.ndarray(names_trn.shape, str)
names_trn_small = np.ndarray(names_trn.shape, str)

large_tst_count = 0
small_tst_count = 0
tst_count = 0
while tst_count < len(spread_tst):
    if abs(spread_tst[tst_count]) >= 5:
        X_tst_linear_large[large_tst_count,:] = X_tst_linear[tst_count,:]
        spread_tst_large[large_tst_count] = spread_tst[tst_count]
        cover_tst_large[large_tst_count] = cover_tst[tst_count]
        names_tst_large[large_tst_count] = names_tst[tst_count]
        large_tst_count+=1
    else:
        X_tst_linear_small[small_tst_count,:] = X_tst_linear[tst_count,:]
        spread_tst_small[small_tst_count] = spread_tst[tst_count]
        cover_tst_small[small_tst_count] = cover_tst[tst_count]
        names_tst_small[small_tst_count] = names_tst[tst_count]
        small_tst_count+=1
    tst_count += 1

large_trn_count = 0
small_trn_count = 0
trn_count = 0
while trn_count < len(spread_trn):
    if abs(spread_trn[trn_count]) >= 5:
        X_trn_linear_large[large_trn_count,:] = X_trn_linear[trn_count,:]
        spread_trn_large[large_trn_count] = spread_trn[trn_count]
        cover_trn_large[large_trn_count] = cover_trn[trn_count]
        names_trn_large[large_trn_count] = names_trn[trn_count]
        large_trn_count+=1
    else:
        X_trn_linear_small[small_trn_count,:] = X_trn_linear[trn_count,:]
        spread_trn_small[small_trn_count] = spread_trn[trn_count]
        cover_trn_small[small_trn_count] = cover_trn[trn_count]
        names_trn_small[small_trn_count] = names_trn[trn_count]
        small_trn_count+=1
    trn_count += 1


X_trn_linear_large = X_trn_linear_large[:large_trn_count,:]
spread_trn_large = spread_trn_large[:large_trn_count]
cover_trn_large = cover_trn_large[:large_trn_count]
names_trn_large = names_trn_large[:large_trn_count]
X_tst_linear_large = X_tst_linear_large[:large_tst_count,:]
spread_tst_large = spread_tst_large[:large_tst_count]
cover_tst_large = cover_tst_large[:large_tst_count]
names_tst_large = names_tst_large[:large_tst_count]
X_trn_linear_small = X_trn_linear_small[:small_trn_count,:]
spread_trn_small = spread_trn_small[:small_trn_count]
cover_trn_small = cover_trn_small[:small_trn_count]
names_trn_small = names_trn_small[:small_trn_count]
X_tst_linear_small = X_tst_linear_small[:small_tst_count,:]
spread_tst_small = spread_tst_small[:small_tst_count]
cover_tst_small = cover_tst_small[:small_tst_count]
names_tst_small = names_tst_small[:small_tst_count]

pandas.DataFrame(X_trn_linear).to_csv("trn_gen.csv")
print(np.any(np.isnan(X_trn_linear)))
print(np.all(np.isfinite(X_trn_linear)))
print(np.any(np.isnan(cover_trn)))
print(np.all(np.isfinite(cover_trn)))

def create_model():
  model=Sequential()

  model.add(Dense(128, activation='relu'))

  model.add(Dropout(0.5))

  model.add(Dense(1024, activation='relu'))

  model.add(Dropout(0.5))

  model.add(Dense(512,activation='relu'))

  model.add(Dropout(0.5))

  model.add(Dense(128,activation='relu'))

  model.add(Dense(2, activation='softmax'))

  return model


batch_size = 128
epochs = 400
class_large = create_model()
class_small = create_model()
class_gen = create_model()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
class_large.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
class_small.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
class_gen.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

early_stopping_ital=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

filepath_ital="top_model_ital.h5"

checkpoint_ital = callbacks.ModelCheckpoint(filepath_ital, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list_ital = [early_stopping_ital,checkpoint_ital]

early_stopping_noital=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

filepath_noital="top_model_noital.h5"

checkpoint_noital = callbacks.ModelCheckpoint(filepath_noital, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list_noital = [early_stopping_noital,checkpoint_noital]


class_large.fit(X_trn_linear_large, cover_trn_large, 
          validation_split=.3,
          shuffle=True,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list_ital)

class_small.fit(X_trn_linear_small, cover_trn_small, 
          validation_split=.3,
          shuffle=True,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list_noital)

class_gen.fit(X_trn_linear, cover_trn, 
          validation_split=.3,
          shuffle=True,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list_noital)





