import os
import cv2
import numpy as np
from tqdm import tqdm

# a function that applies zoom blur to an image
def zoom_blur_2d(BASE_PATH_RGB, BASE_TARGET_PATH):
    iterations = 5
    severity = 2 
    blur = 0.01*severity
    # read the image
    img = cv2.imread(BASE_PATH_RGB)
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
    cv2.imwrite(BASE_TARGET_PATH, img)
    
