from prepare_data import *
from viola_jones import *
import numpy as np
import json


# parameters
cartoons = ["cartoon_images", "real"]
max_width = 700
rgb2gray = True
use_combined_detection = False
save_faces = True

# classifiers
face_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_eye.xml")


results = {}

for cartoon in cartoons:

    # paths
    path_dir_test = f"../data/test/{cartoon}"
    path_dir_output = f"../output/{cartoon}"
    if use_combined_detection:
        path_dir_output = f"{path_dir_output}_2"
  

    # preprocessing
    preprocessing(max_width, rgb2gray, cartoon)
    dir_structure(path_dir_output)
   

    for filename in os.listdir(path_dir_test):
        img = cv2.imread(f"{path_dir_test}/{filename}")
        if img is not None:
            if use_combined_detection:
                faces = viola_jones_combine(img, face_cascade, eye_cascade)
            else:
                faces = viola_jones_face(img, face_cascade)

            
            if faces is not None:
                if save_faces:
                    for (x, y, w, h) in faces:
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), (4, 8, 170), 12)
            if save_faces:
                cv2.imwrite(f"{path_dir_output}/{filename}", img)

         


