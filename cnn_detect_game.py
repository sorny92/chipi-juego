import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from extra_function import *
from keras.models import load_model
from scipy.ndimage.measurements import label


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
        cars_image = np.copy(frame)
        #image = find_cars(image,(400,656), 1, svc, X_scaler, 9, 8, 2,(16, 16), 16)
        image, heatmap = detect_car(frame,(400,656))
        labels = label(heatmap)
        cars_image = draw_labeled_bboxes(cars_image, labels)
#        plt.imshow(img)
#        plt.pause(0.01)
        writer.writeFrame(cars_image)
    writer.close()

def detect_car(image):
    heatmap = np.zeros((image[:,:,0].shape))
    heat_boxes = []
    xy_window = (24, 24)
    lower = np.array([75,130,140], dtype = "uint8")
    upper = np.array([85,144,150], dtype = "uint8")

    mask = cv2.inRange(image, lower, upper)
    boundary_image = cv2.bitwise_and(image, image, mask=mask)
    boundary_image = cv2.dilate(boundary_image, np.ones((5,5),np.uint8), iterations = 1)
    boundary_image = cv2.erode(boundary_image, np.ones((7,7),np.uint8), iterations = 1)
    boundary_image = cv2.dilate(boundary_image, np.ones((62,62),np.uint8), iterations = 1)
    boundary_image = cv2.erode(boundary_image, np.ones((60,60),np.uint8), iterations = 1)
    #boundary_image = cv2.Canny(boundary_image, 60,100)
    boundary_image = np.float32(cv2.cvtColor(boundary_image, cv2.COLOR_BGR2GRAY))
    boundary_image = cv2.cornerHarris(boundary_image, 10, 9, 0.04)
    aux = np.copy(boundary_image)
    aux[boundary_image<250]=0
    aux[boundary_image<200]=255
    boundary_image = np.hstack((boundary_image, aux))
    boundary_image = cv2.dilate(boundary_image,None)
    # Loop over the image in vertical bars of width x
    #take the smallest and the higest positions wich are 255
    #do the same in horizontal bars of height x
    # the smalles of both is the point1 and the highest of both is the point2
    windows = slide_window(image, xy_window=xy_window, xy_overlap=(0, 0))
    image_analize = np.copy(image)
    '''for window in windows:
        window_image = image_analize[window[0][1]:window[1][1],window[0][0]:window[1][0]]
        if(window_image.shape[0] >= xy_window[0] and window_image.shape[1] >= xy_window[1]):
            window_image = cv2.resize(window_image, (24, 24))
            test_prediction = model.predict(window_image[None, :, :, :])
            print (test_prediction)
            if test_prediction != 0:
                image = cv2.rectangle(image,(window[0][0], window[0][1]),(window[1][0],window[1][1]),(0,0,255),2) 
                heat_boxes.append([(window[0][0], window[0][1]),(window[1][0],window[1][1])])
    heatmap = add_heat(heatmap, heat_boxes)
    heatmap = apply_threshold(heatmap, 1)'''
    image = draw_boxes(image, windows, color=(0, 255, 0), thick=2)
    return boundary_image, heatmap

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def test_images(origin):
    image_names = glob.glob(origin)
    i=0
    for name in image_names:
        print('Image ', i)
        i += 1
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cars_image = np.copy(image)
        #image = find_cars(image,(400,656), 1, svc, X_scaler, 9, 8, 2,(16, 16), 16)
        image, heatmap = detect_car(image)
        labels = label(heatmap)
        cars_image = draw_labeled_bboxes(cars_image, labels)
        plt.imshow(image, cmap='gray')
        plt.pause(50)

test_images('/home/esteve/chipi-juego/examples/WhatsApp Image 2017-09-13 at 22.03.23(1).jpeg')
#process_video('project_video.mp4')