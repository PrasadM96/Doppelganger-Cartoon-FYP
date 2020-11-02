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
    
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    
    # Let us print the no. of faces found
    print('Faces found: ', len(faces_rect))
    
    if(len(faces_rect)==0):
        return []
    
    
    a=0
    
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy,  (x-a,y-a),   (x+w+a,y+h+a), (0, 255, 0), 8)

    return image_copy
    
 
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_paths = get_image_paths('man_20');
# print(image_paths)
# Display all images inside image_paths
all=0
detected=0
for imagePath in image_paths:

    #loading image
    test_image = cv2.imread(imagePath)
    all=all+1
#     display image
    f,axarr = plt.subplots(1,2)
    plt.axis('off')
    axarr[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))

    #call the function to detect faces
    faces = detect_faces(haar_cascade_face, test_image)

    #convert to RGB and display image
    if(len(faces)!=0):
        detected=detected+1
        axarr[1].imshow(convertToRGB(faces))

print('{} out of {}'.format(all,detected))
