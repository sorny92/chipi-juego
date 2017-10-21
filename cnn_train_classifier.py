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

images_background = glob.glob('./images/background/*')
images_background_blue = glob.glob('./images/background_blue/*')
images_ball = glob.glob('./images/ball/*')
images_box = glob.glob('./images/box/*')
images_hole = glob.glob('./images/hole/*')
images_rock = glob.glob('./images/rock/*')
background_features = []
background_blue_features = []
ball_features = []
box_features = []
hole_features = []
rock_features = []
images = []
labels = []
for name in images_background:
    image = cv2.imread(name)
    image = cv2.resize(image, (24,24))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    background_features.append(image)
    labels.append(0)

for name in images_background_blue:
    image = cv2.imread(name)
    image = cv2.resize(image, (24,24))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    background_blue_features.append(image)
    labels.append(1)
    
for name in images_ball:
    image = cv2.imread(name)
    image = cv2.resize(image, (24,24))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ball_features.append(image)
    labels.append(2)
for name in images_box:
    image = cv2.imread(name)
    image = cv2.resize(image, (24,24))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    box_features.append(image)
    labels.append(3)
for name in images_hole:
    image = cv2.imread(name)
    image = cv2.resize(image, (24,24))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hole_features.append(image)
    labels.append(4)
for name in images_rock:
    image = cv2.imread(name)
    image = cv2.resize(image, (24,24))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rock_features.append(image)
    labels.append(5)
from keras.utils.np_utils import to_categorical
y = to_categorical(labels)
print(len(background_features))
print(len(background_blue_features))
print(len(ball_features))
print(len(box_features))
print(len(hole_features))
print(len(rock_features))

X = np.vstack((background_features, background_blue_features,
              ball_features, box_features, hole_features, rock_features)).astype(np.float64)
print(X.shape)
print(y.shape)
X, y = sklearn.utils.shuffle(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

BATCH_SIZE = 12
EPOCHS = 50

ch, row, col = 3, 24, 24

def preprocess(x):
    x = x/127.5 - 1
    return x
    
model = Sequential()
model.add(Lambda(preprocess, input_shape=(row, col, ch)))

model.add(Conv2D(24, (5, 5)))
model.add(Activation('relu'))

model.add(Conv2D(36, (5, 5)))
model.add(Activation('relu'))

model.add(Conv2D(48, (3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), strides=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(60))
model.add(Dropout(0.4))
model.add(Dense(20))
model.add(Dropout(0.4))
model.add(Dense(6))
model.add(Activation('relu'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print (score)
#print (model.predict(X_test[1,:]))
model.save('model.h5')