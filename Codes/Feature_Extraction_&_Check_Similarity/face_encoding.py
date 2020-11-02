from PIL import Image
import PIL.Image
import dlib
import numpy as np
from PIL import ImageFile
import face_recognition_models
import cv2
import os
import json
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

ImageFile.LOAD_TRUNCATED_IMAGES = True

d = "mmod_human_face_detector.dat"
p68 = "shape_predictor_68_face_landmarks.dat"
p5 = "shape_predictor_5_face_landmarks.dat"
e = "dlib_face_recognition_resnet_model_v1.dat"

face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(d)
pose_predictor_68_point = dlib.shape_predictor(p68)
pose_predictor_5_point = dlib.shape_predictor(p5)
face_encoder = dlib.face_recognition_model_v1(e)

image_formats = ["png", "jpg"]; # Let suppose we want to display png & jpg images (specify more if you want)

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
            image_paths = glob.glob(os.path.join(current_dir, file, "*." + image_format))
            if image_paths:
                paths.extend(image_paths);

    return paths
    
def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)
 
def face_distance(face_encodings, face_to_compare, method="euclidean"):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    if method=="manhattan":
        return np.linalg.norm(face_encodings - face_to_compare, ord=1, axis=1)
    else:
        return np.linalg.norm(face_encodings - face_to_compare, axis=1)
        
def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)
    
def _raw_face_locations(img, number_of_times_to_upsample=1, detector="cnn"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if detector == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)
        
def face_locations(img, number_of_times_to_upsample=1, detector="cnn"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    if detector == "cnn":
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]
    else:
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]

def _raw_face_landmarks(face_image, model="large", detector="cnn"):

    face_locations = _raw_face_locations(face_image,1,detector)
    
#     for face_location in face_locations:
#         x = face_location.rect.top()
#         y = face_location.rect.right()
#         x1 = face_location.rect.bottom()
#         y1= face_location.rect.left()
#         cv2.rectangle(face_image, (x,y), (x1,y1), (0, 255, 0), 8)
#     plt.figure()
#     plt.imshow(face_image)

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    if detector == "cnn":
        return [pose_predictor(face_image, face_location.rect) for face_location in face_locations]
    else:
        return [pose_predictor(face_image, face_location) for face_location in face_locations]

def face_landmarks(face_image, model="large", detector="cnn"):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

    :param face_image: image to search
    :param face_locations: Optionally provide a list of face locations to check.
    :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    landmarks = _raw_face_landmarks(face_image, model, detector)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    if model == 'large':
        return [{
            "chin": points[0:17],
            "left_eyebrow": points[17:22],
            "right_eyebrow": points[22:27],
            "nose_bridge": points[27:31],
            "nose_tip": points[31:36],
            "left_eye": points[36:42],
            "right_eye": points[42:48],
            "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
            "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
        } for points in landmarks_as_tuples]
    elif model == 'small':
        return [{
            "nose_tip": [points[4]],
            "left_eye": points[2:4],
            "right_eye": points[0:2],
        } for points in landmarks_as_tuples]
    else:
        raise ValueError("Invalid landmarks model type. Supported models are ['small', 'large'].")

def face_encodings(face_image, num_jitters=1, model="large", detector="cnn"):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, model, detector)

    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]
    
def show(face_locations,Img):

    # Print the location of each face in this image
    top, right, bottom, left = face_locations[0]
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = Img[top:bottom, left:right]
    #face_image = Img[(top-5):(bottom+5), (left-5):(right+5)]                    
    face_image = cv2.resize(face_image, (224, 224))
    pil_image = Image.fromarray(face_image)
    pil_image.show()
    return 0

image_paths = get_image_paths("repo3");


# Display all images inside image_paths
for i in range(0,len(image_paths),2):
    # Load the jpg file into a numpy array
    img = load_img(image_paths[i])
    img2 = load_img(image_paths[i+1])
    img = np.array(img)
    img2 = np.array(img2)
    if(len(face_encodings(img))!=0 and len(face_encodings(img2))!=0):
        img1_face_encoding = face_encodings(img)[0]
        print(img_face_encoding.shape)
        img2_face_encoding = face_encodings(img2)[0]
        manhattan_distance = face_distance([img1_face_encoding], img2_face_encoding, method="manhattan")
        euclidean_distance = face_distance([img1_face_encoding], img2_face_encoding)
        print("Distances between {} and {}".format(image_paths[i],image_paths[i+1]))
        print("Manhattan distance : {}".format(manhattan_distance))
        print("Euclidean distance : {}".format(euclidean_distance))

'''
img = load_img('repo3/Agnes_Gru/cartoon.jpg')
img2 = load_img('repo3/Agnes_Gru/real.jpg')
img = np.array(img)
img2 = np.array(img2)
#display image
f,axarr = plt.subplots(1,2, figsize=(5,5))
axarr[0].imshow(img)
axarr[1].imshow(img2)
img1_face_encoding = face_encodings(img)[0]
img2_face_encoding = face_encodings(img2)[0]
manhattan_distance = face_distance([img1_face_encoding], img2_face_encoding, method="manhattan")
euclidean_distance = face_distance([img1_face_encoding], img2_face_encoding)
print("Distances between {} and {}".format(image_paths[i],image_paths[i+1]))
print("Manhattan distance : {}".format(manhattan_distance))
print("Euclidean distance : {}".format(euclidean_distance))
'''
