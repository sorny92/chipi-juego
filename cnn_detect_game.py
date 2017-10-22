import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from extra_function import *
from keras.models import *
from scipy.ndimage.measurements import label
import os
import random
from PIL import Image

# ONLY CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

json_string = open('model.json','r').read()
model = model_from_json(json_string)
model.load_weights('model.h5')

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def detect_objects(image, return_image=False):
    colors = [(0,0,0), (0,0,125), (200,200,0), (255,255,255), (0,255,0), (0,255,255)]

    # Threshold to get the outer color of the game
    lower = np.array([75,130,140], dtype = "uint8")
    upper = np.array([85,144,150], dtype = "uint8")
    mask = cv2.inRange(image, lower, upper)

    # Create binay mask
    boundary_image = cv2.bitwise_and(image, image, mask=mask)

    # Cleaning up noise from the image with a serio of dilations and erosions
    boundary_image = cv2.dilate(boundary_image, np.ones((5,5),np.uint8), iterations = 1)
    boundary_image = cv2.erode(boundary_image, np.ones((7,7),np.uint8), iterations = 1)
    boundary_image = cv2.dilate(boundary_image, np.ones((62,62),np.uint8), iterations = 1)
    boundary_image = cv2.erode(boundary_image, np.ones((60,60),np.uint8), iterations = 1)
    boundary_image = np.float32(cv2.cvtColor(boundary_image, cv2.COLOR_BGR2GRAY))

    #Find the corners boundaries of the gameboard
    boundary_image = cv2.cornerHarris(boundary_image, 10, 9, 0.04)
    aux = np.copy(boundary_image)
    aux[boundary_image<250]=0
    aux[boundary_image<200]=255
    boundary_image = cv2.dilate(boundary_image,None)

    # Get the coordinates of the gameboard corners
    points = np.argwhere(boundary_image>100)
    p1 = [min(points[:,0]),min(points[:,1])]
    p2 = [max(points[:,0]),max(points[:,1])]

    print (p1, p2)

    # Cut the gameboard in a 15x15 matrix
    xy_window = ((p2[0]-p1[0])/15, 
                 (p2[0]-p1[0])/15)

    # Only split the gameboard
    x_start_stop = [p1[1],p2[1]]
    y_start_stop = [p1[0],p2[0]]

    #Generate the windows
    windows = slide_window(image, 
            x_start_stop=x_start_stop,
            y_start_stop=y_start_stop,
            xy_window=xy_window, 
            xy_overlap=(0, 0))
    image_analize = np.copy(image)
    results = [[]]
    row = 0
    column = 0
    for window in windows:
        window_image = image_analize[window[0][1]:window[1][1],window[0][0]:window[1][0]]
        if(window_image.shape[0] >= xy_window[0] and window_image.shape[1] >= xy_window[1]):
            window_image = cv2.resize(window_image, (24, 24))
            test_prediction = model.predict(window_image[None, :, :, :])
            #print test_prediction, window
            if return_image:
                c = colors[np.argmax(test_prediction)]
                image_analize = cv2.rectangle(image_analize, window[0], window[1], c, -1)
            else:
                results[row].append(np.argmax(test_prediction))
                column += 1
            if not column % 15:
                results.append([])
                row += 1
                column = 0
    if return_image:
        image_analize = draw_boxes(image_analize, windows, color=(0, 255, 0), thick=2)
        return image_analize
    else:
        results.pop()
        return results

def test_images(origin):
    image_names = glob.glob(origin)
    i=0
    for name in image_names:
        print('Image ', i)
        i += 1
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print (detect_objects(image, False))
        #plt.imshow(image, cmap='gray')
        #plt.pause(50)

test_images('/home/esteve/chipi-juego/examples/*')