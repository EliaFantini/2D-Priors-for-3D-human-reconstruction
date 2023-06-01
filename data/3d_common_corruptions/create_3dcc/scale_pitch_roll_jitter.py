import os
import cv2
import numpy as np
from tqdm import tqdm

# a function that applies zoom blur to an image
def zoom_blur_2d(BASE_PATH_RGB, BASE_TARGET_PATH):
    
    
    iterations = 5

    severity = 2
    TARGET_PATH = os.path.join(BASE_TARGET_PATH, "zoom_blur_2d", str(severity))
    if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)
    blur = 0.01*severity
    list_of_images = os.listdir(BASE_PATH_RGB)
    for image_path in list_of_images:
        # read the image
        img = cv2.imread(os.path.join(BASE_PATH_RGB, image_path))

        w, h = img.shape[:2]
        center_x = w / 2
        center_y = h / 2
        growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
        shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
        growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
        shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
        for i in range(iterations):
            tmp1 = cv2.remap(img, growMapx, growMapy, cv2.INTER_LINEAR)
            tmp2 = cv2.remap(img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
            img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
        cv2.imwrite(os.path.join(TARGET_PATH, image_path), img)
    

# a function that takes as input a path to a folder containing the original images and a path to a folder containing the target images and zooms into the original images and saves the resulting images in the target folder
def apply_scale(BASE_PATH_RGB, BASE_TARGET_PATH):
        
        # create the target folder if it does not exist
        TARGET_PATH = os.path.join(BASE_TARGET_PATH, "scale")
        if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)
        
        # define the scale factors
        scale_factors = [1.5,2.0,2.5]
        
        # iterate over the scale factors
        for scale_factor in tqdm(scale_factors):
            # create the target folder if it does not exist
            TARGET_PATH = os.path.join(BASE_TARGET_PATH, "scale", str(scale_factor))
            if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)
            
            # iterate over the images in the folder
            list_of_images = os.listdir(BASE_PATH_RGB)
            for image_path in list_of_images:
                # read the image
                img = cv2.imread(os.path.join(BASE_PATH_RGB, image_path))
                
                # zoom the image
                img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                
                # save the image
                cv2.imwrite(os.path.join(TARGET_PATH, image_path), img)

# a function that takes as input a path to a folder containing the original images and a path to a folder containing the target images and pitch the original images and saves the resulting images in the target folder
def apply_pitch(BASE_PATH_RGB, BASE_TARGET_PATH):
            
            # create the target folder if it does not exist
            TARGET_PATH = os.path.join(BASE_TARGET_PATH, "pitch")
            if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)
            
            # define the pitch factors
            pitch_factors = [-10, -5, 5, 10]
            
            # iterate over the pitch factors
            for pitch_factor in tqdm(pitch_factors):
                # create the target folder if it does not exist
                TARGET_PATH = os.path.join(BASE_TARGET_PATH, "pitch", str(pitch_factor))
                if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)
                
                # iterate over the images in the folder
                list_of_images = os.listdir(BASE_PATH_RGB)
                for image_path in list_of_images:
                    # read the image
                    img = cv2.imread(os.path.join(BASE_PATH_RGB, image_path))
                    
                    # get the image size
                    h, w = img.shape[:2]
                    
                    # define the rotation matrix
                    M = cv2.getRotationMatrix2D((w/2, h/2), pitch_factor, 1)
                    
                    # rotate the image
                    img = cv2.warpAffine(img, M, (w, h))
                    
                    # save the image
                    cv2.imwrite(os.path.join(TARGET_PATH, image_path), img)

# a function that takes as input a path to a folder containing the original images and a path to a folder containing the target images and roll the original images and saves the resulting images in the target folder
def apply_roll(BASE_PATH_RGB, BASE_TARGET_PATH):
            # define the roll factor 
            roll_factor = 80
            # read the image
            img = cv2.imread(BASE_PATH_RGB)
            img_shape = img.shape
            
            # get the image size
            h, w = img_shape[:2]
            
            # define the rotation matrix
            M = cv2.getRotationMatrix2D((w/2, h/2), roll_factor, 1)
            
            # rotate the image
            img = cv2.warpAffine(img, M, (w, h))
            
            # save the image
            cv2.imwrite(BASE_TARGET_PATH, img)
    
    