import cv2
import numpy as np
import glob

image_names = glob.glob('./test_images/*')

for name in image_names:
    image = cv2.imread(name)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def train_color_classifier():
    pass

def train_hog_classifier():
    pass


