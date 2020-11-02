from PIL import Image
import numpy as np
import face_recognition
import cv2
import os
import json
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Model
from pickle import dump
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import csv
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
%matplotlib inline

image_formats = ["png", "jpg"]; # Let suppose we want to display png & jpg images (specify more if you want)

def get_image_paths(current_dir):
    files = os.listdir(current_dir);
    paths = []; # To store relative paths of all png and jpg images

    for file in files:
        file = file.strip()
        for image_format in image_formats:
            image_paths = glob.glob(os.path.join( current_dir,file,"*." + image_format))
            if image_paths:
                paths.extend(image_paths);

    return paths

    
image_paths = get_image_paths('man_20');
print(image_paths)
# Display all images inside image_paths
all2=0
detected=0
for imagePath in image_paths:

    img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB) 
#     plt.figure()
#     plt.imshow(img)
    detector = MTCNN()
    detector=detector.detect_faces(img)
    all2 = all2+1
    if(len(detector)!=0):
        face_locations = detector[0]['box']
        a=30
        x, y, w, h = face_locations
        
        detected=detected+1

        cv2.rectangle(img,  (x,y),   (x+w,y+h), (0, 255, 0), 3)

        plt.figure()
        plt.axis('off')
        plt.imshow(img)
