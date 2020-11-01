#  Face detection using Viola-Jones

We observe how the Viola-Jones face detection algorithm performs on "non-human human" faces, i.e. cartoon characters. Our data consists of cartoon images and real human images. Here we  try to enhance the detector by combining the Viola-Jones face and eye classifier.

![Example](example.png)


#### Requirements
- Anaconda Python Distribution, Python 3.7.4, installed from: [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/).
- R version 3.6.1, installed from: [https://www.r-project.org/](https://www.r-project.org/).

#### Environment setup
- Setup the environment using environment.yml file you find in the directory: `conda env create -f environment.yml`.
- Activate the environment: `conda activate cartoon-face-detection`.


#### Folder structure

    .
    ├── data                    
    │   ├── raw         		# subfolders of cartoon and real human images
    │   └── test                # preprocessed images in  subfolders
    ├── output                  # detection output in subfolders
    ├── src                    	# Python  scripts
    ├── environment.yml			# conda environment file
    ├── example.png
    └── README.md

#### Run experiments
Run `script.py` to obtain detections. The script will take care of preprocessing, detection and will also evaluate the detections. 

Before running the script, user can change the following parameters:  

- `cartoons` **list** List of cartoons we want to use in our experiment.  
- `max_width` **integer** Maximum width for the input images in px or `None` if you don't want to resize the images.  
- `rgb2gray` **boolean** True if you want to convert the input images to grayscale.   
- `use_combined_detection` **boolean** True if you want to use our combined VJ detector.  
- `save_faces` **boolean** True if you want to draw the detections on the images and save them.  

