import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from extra_function import *
import skvideo.io
from keras.models import load_model


model = load_model('model.h5')

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

def process_video(video):
    vid = skvideo.io.vreader(video)
    writer = skvideo.io.FFmpegWriter('{}_output.mp4'.format(video), verbosity=1)
    for frame in vid:
        #output = find_cars(frame,(400,656), 1, svc, X_scaler, 9, 8, 2,(16, 16), 16)
        output = detect_car(frame, (400,656))
#        plt.imshow(img)
#        plt.pause(0.01)
        writer.writeFrame(output)
    writer.close()

def detect_car(image, y_start_stop):
    xy_window = (128, 128)
    windows = slide_window(image, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=(0.7, 0.7))
    xy_window = (64, 64)
    windows += slide_window(image, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=(0.5, 0.5))
    image_analize = np.copy(image)
    for window in windows:
        window_image = image_analize[window[0][1]:window[1][1],window[0][0]:window[1][0]]
        if(window_image.shape[0] >= xy_window[0] and window_image.shape[1] >= xy_window[1]):
            if (window_image.shape[0] > 64 or window_image.shape[1] > 64):
                window_image = cv2.resize(window_image, (64, 64))
            test_prediction = model.predict(window_image[None, :, :, :])
            #print(test_prediction)
            if test_prediction != 0:
                image = cv2.rectangle(image,(window[0][0], window[0][1]),(window[1][0],window[1][1]),(0,0,255),2) 
    #image = draw_boxes(image, windows, color=(0, 255, 0), thick=2)
    return image

def test_images(origin):
    image_names = glob.glob(origin)
    i=0
    for name in image_names:
        print('Image ', i)
        i += 1
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = find_cars(image,(400,656), 1, svc, X_scaler, 9, 8, 2,(16, 16), 16)
        image = detect_car(image,(400,656))
        plt.imshow(image)
        plt.pause(2)

#test_images('./test_images/*')
process_video('project_video.mp4')