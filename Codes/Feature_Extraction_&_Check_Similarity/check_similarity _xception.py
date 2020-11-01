

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
from tensorflow.keras.applications import Xception 
from keras.models import Model
from pickle import dump
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import csv

image_formats = ["png", "jpg","jpeg"]; # Let suppose we want to display png & jpg images (specify more if you want)

def show_images(image_file_name):
    print("Displaying ", image_file_name)
    img=mpimg.imread(image_file_name)
    imgplot = plt.imshow(img)
    plt.show()

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

def normalizeFace(face_locations,Img):

     # Print the location of each face in this image
    top, right, bottom, left = face_locations[0]
    #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    
    a=30
    # for (top, right, bottom,left) in face_locations:
    #     cv2.rectangle(Img,  (top,right),   (bottom,left), (0, 255, 0), 8)
    
    # plt.figure()
    # plt.imshow(Img)
        
    # You can access the actual face itself like this:
    face_image = Img[top:bottom, left:right]
    #face_image = image[(top-10):(bottom+10), (left-10):(right+10)]                    
    face_image = cv2.resize(face_image, (224, 224))
    pil_image = Image.fromarray(face_image)
    #normalize in [-1, +1]
    img_pixels = image.img_to_array(face_image)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 127.5
    img_pixels -= 1
    return img_pixels

def findManhattanDistance(source_representation, test_representation):
    return np.linalg.norm(source_representation - test_representation, ord=1)
 
def findEuclideanDistance(source_representation, test_representation):
    return np.linalg.norm(source_representation - test_representation)

def verifyFace(img1, img2):
    manhattan_distance = findManhattanDistance(img1, img2)
    euclidean_distance = findEuclideanDistance(img1, img2)
    print("Manhattan distance : {}".format(manhattan_distance))
    print("Euclidean distance : {}".format(euclidean_distance))



image_paths = get_image_paths('/content/drive/My Drive/CO421-Project-images/aaaaaaaaaaaaaaaaaaaaaaaa');
# Display all images inside image_paths
for i in range(0,len(image_paths),2):
    # Load the jpg file into a numpy array
    
    img = load_img(image_paths[i])
    img2 = load_img(image_paths[i+1])
    img = np.array(img)
    img2 = np.array(img2)
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")
    face_locations2 = face_recognition.face_locations(img2, number_of_times_to_upsample=0, model="cnn")
    if((len(face_locations)!=1) or (len(face_locations2)!=1)):
        print("Found {} faces in this photograph. Please use photograph only one face included".format(len(face_locations)))
    else:
        img_pixels = normalizeFace(face_locations,img)
        img_pixels2 = normalizeFace(face_locations2,img2)
       
     
        #VGG16
        model = VGG16(weights='imagenet')
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        features = model.predict(img_pixels)
        print(features.shape)
        features2 = model.predict(img_pixels2)
        print("Distances between {} and {}".format(image_paths[i],image_paths[i+1]))
        verifyFace(features[0],features2[0])

