import os
import csv
import cv2
import numpy as np
import sklearn
import glob

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, Activation, Lambda, Cropping2D, GaussianNoise
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

image_names_car = glob.glob('./vehicles/*/*')
image_names_no_car = glob.glob('./no-car64/*')

car_features = []
no_car_features = []

print('loading images 1')
for name in image_names_car:
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    car_features.append(image)
print('loading images 2')
for name in image_names_no_car:
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    no_car_features.append(image)
print('Loaded')

car_features = sklearn.utils.shuffle(car_features)
car_features = car_features[:5000]
y = np.hstack((np.ones(len(car_features)),np.zeros(len(no_car_features))))
X = np.vstack((car_features, no_car_features)).astype(np.float64)
print(X[0].shape)
X, y = sklearn.utils.shuffle(X, y)

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

def generator(x, y, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            X_batch = x[offset:offset+batch_size]
            y_batch = y[offset:offset+batch_size]
            yield sklearn.utils.shuffle(X_batch, y_batch)

BATCH_SIZE = 32
EPOCHS = 3

train_generator = generator(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = generator(X_test, y_test, batch_size=BATCH_SIZE)

ch, row, col = 3, 64, 64

def preprocess(x):
    x = x/127.5 - 1
    return x
    
model = Sequential()
model.add(Lambda(preprocess, input_shape=(row, col, ch)))

model.add(Conv2D(24, (5, 5)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(36, (5, 5)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(48, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(50))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(loss='mse', optimizer='adam')
model.fit(X, y, batch_size = BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
model.save('model_v2.h5')