import cv2
import numpy as np
import glob
import time
import pickle
import os.path
import matplotlib.pyplot as plt

from extra_function import *
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

image_names_car = glob.glob('./vehicles/GTI*/*')
image_names_no_car = glob.glob('./no-car64/*')
print('Images with cars: {}'.format(len(image_names_car)))
print('Images without cars: {}'.format(len(image_names_no_car)))
car_features = []
no_car_features = []
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (16, 16)
hist_bins = 16

car_features_file = './car_features.pkl'
if(os.path.isfile(car_features_file)):
    print('Loading car features from file...')
    with open(car_features_file, 'rb') as f:
        car_features = pickle.load(f)
else:
    print('Generating car features from file...')
    for name in image_names_car:
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hog_features = []
        for channel in range(image.shape[2]):
            hog_features.append(get_hog_features(image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
        spatial_features = bin_spatial(image, size=spatial_size)
        hist_features = color_hist(image, nbins=hist_bins)
        features = np.hstack((spatial_features, hist_features, hog_features))
        #print(features.shape)
        car_features.append(features)
    with open(car_features_file, 'wb') as f:
        pickle.dump(car_features, f)
        print('Generated')

no_car_features_file = './no_car_features.pkl'
if(os.path.isfile(no_car_features_file)):
    print('Loading no_car features from file...')
    with open(no_car_features_file, 'rb') as f:
        no_car_features = pickle.load(f)
else:
    print('Generating no_car features from file...')
    for name in image_names_no_car:
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hog_features = []
        for channel in range(image.shape[2]):
            hog_features.append(get_hog_features(image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
        spatial_features = bin_spatial(image, size=spatial_size)
        hist_features = color_hist(image, nbins=hist_bins)
        features = np.hstack((spatial_features, hist_features, hog_features))
        no_car_features.append(features)
    with open(no_car_features_file, 'wb') as f:
        pickle.dump(no_car_features, f)
        print('Generated')

y = np.hstack((np.ones(len(car_features)),np.zeros(len(no_car_features))))
X = np.vstack((car_features, no_car_features)).astype(np.float64)
X, y = shuffle(X, y)

#X = X[:2000]
#y = y[:2000]
print(X.shape)
X_scaler = StandardScaler().fit(X)
with open('X_scaler.pkl', 'wb') as f:
    pickle.dump(X_scaler, f)
X = X_scaler.transform(X)
print([X[0]])

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
with open('linearSVC.model', 'wb') as f:
    pickle.dump(svc, f)
    print('Model generated & saved')
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))