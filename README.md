# Vehicle Detection
[![CarND-Vehicle-Detection - SDC](http://i.makeagif.com/media/7-03-2017/fahV03.gif)](https://youtu.be/cdRn3OD5VT0)   
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## [Here you can find the Writeup](https://github.com/sorny92/CarND-VehicleDetection/blob/master/writeup.md)

Files in the project
---
* `cnn_detect_car.py` Script to detect cars in frames or videos using the model.h5 which is a Convolutional neural network.
* `cnn_train_classifier.py` File to train the CNN classifier.
* `detect_car.py` Same as `cnn_detect_car.py` but using only computer vision methods.
* `extra_function.py` This is a library with some methods the program use.
* `linearSVC.model` This is the SVC model trained to detect cars using computer vision methods.
* `model.h5` Trained model for the CNN version
* `model_v2.h5` Another model for the CNN version that works differently
* `no_car_generator.py` Script to generate images from videos to generate more data values that doesn't contain cars on it.
* `train_classifier.py` Script used to train the SVC.
* `writeup.md` Here you can find a description of the process I took to develop this solution

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
