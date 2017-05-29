# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[svcscreen]: ./output_images/svc.png
[cnnscreen]: ./output_images/cnn.png
[model]: ./output_images/model.png
[figure1]: ./output_images/figure_1.png
[figure2]: ./output_images/figure_2.png
[figure3]: ./output_images/figure_3.png
[figure4]: ./output_images/figure_4.png
[figure5]: ./output_images/figure_5.png
[figure6]: ./output_images/figure_6.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file train_classifer.py. Between the lines 17 and 40. Where you can see two methods. The first one `get_hog_features` which has the call to the hog method from sklearn. 
Then the method `generate_hog_features` which takes the frames do the next:
* convert them to YCrCb color space
* Get hog features for every channel
* Append them in an array
* Get spatial features 
* Get histogram features
* stack them in one array

With this two methods, now I can generate the features for all the dataset.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tested different values and the best to detect cars with minimal false positives and a moderate response time where the next:
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* spatial_size = 16x16
* hist_bins = 16

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained the classifier between the lines 98 and 109. It is a basic LinearSVC which I split in a training dataset of 0.7

### Convolution Neural Network as classifier

I studied another aproach to recognice the cars which involved training a CNN. I used the same architecture than in the Project 3 to identify the cars in the sliding windows.  
Here you can see the model:  
![Model][model]

This aproach has some good points which is that is more robust with the data that is provided and easy to train because with a simple normalization of the image the model gets quite good results.

Here we can see two images from the same frame. The first one is with the SVC model and the second one with the CNN.

![Svcmodel][svcscreen]
![Svcmodel][cnnscreen]

The CNN during almost all the video is more robust detecting the cars without any kind of filtering except in the white pavement (more data could solve the problem)
The SVC model needs filtering between frames to ease the output of the process.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use a window to detect cars between the pixel 400 and 656 in the vertical axis. This is due because under the 670 you can see part of the car and above the 400 pixels you will not find any car.

Also in the CNN version I apply a multi-scale window system because of this way I can select the cars in different distances and also it helps to get rid of false positives because usually I will get in the same place several matches as car. This will appear in as a hotspot in the heatmap.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  You can see the images in the next part with 6 examples.  

To improve the detection of cars in the CNN version I use a multi-scale and multi-window search which allow me to find the car when it is smaller (located in the foreground).
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result with SVC](./output_images/project_video_SVC.mp4)
Here's a [link to my video result with CNN](./output_images/project_video_CNN.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

This behavour is also applied in time, this means I'm storing the last 3 heatmaps so I can take in account the movement of the cars in time. This fact will helpt me to get rid of false positives because if there is a false positive in a lonely frame it will not appear.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their outputs.
In the top-left corner you find the output of the detection.  
In the top-right corner the output of the heatmap
In the bottom-left you find the label map.
In the bottom-right the output in that frame.

![alt text][figure1]
![alt text][figure2]
![alt text][figure3]
![alt text][figure4]
![alt text][figure5]
![alt text][figure6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Firstly I decided to use only HOG features in the grayscale image which I tough it could work. It gave me an accuracy of the 99% which I though it was wonderful, but it end up generating to much noise in the image so I decided to use HOG features in all the channels.  
Then I tested different color spaces which gave me the higher accuracy, which end up being th YCrCb.
Firstly I was using the KITTI dataset too, but using it didn't help to detect the cars correctly. Then is when I decided to use a deep learning approach. Using all my dataset was able to get the results you can see in the output video of the CNN.

Still with the efforts tuning the parameters with the computer vision approach, using CNN was easier to solve the problem and with better results with the downpoint of the computer power need to process it. But the power of GPUs is increasing really fast which is making possible to detect in real time with architectures as YOLO.

The solution can be improved with easing between the frames for every labeled car. Then reduce the window for detection in the zone around the car is detected.

Then use an initial window in the horizont and in the parts where a car could appear because we know that a car is not going to appear suddenly in the middle of the road so we don't need to check if there is a car in front of us if it wasn't there in the last frame.

