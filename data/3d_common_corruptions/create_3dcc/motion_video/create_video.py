import torch
import torchvision
import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy 
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile
import tqdm
import parse

import pdb
from PIL import Image
import numpy as np

from motion_video.video_distortions import *


import sys
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils import data
from collections import defaultdict
from torch.nn.parallel import parallel_apply

import PIL
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt
import inspect
import natsort
import json
import pdb
import glob
import numpy as np

from dataset import RGBAndDepthDataset

### based on https://github.com/sniklaus/3d-ken-burns and https://github.com/Newbeeyoung/Video-Corruption-Robustness 

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}
CLASS = ''

exec(open('./motion_video/common.py', 'r').read())

exec(open('./motion_video/models/disparity-estimation.py', 'r').read())
exec(open('./motion_video/models/disparity-adjustment.py', 'r').read())
exec(open('./motion_video/models/disparity-refinement.py', 'r').read())
exec(open('./motion_video/models/pointcloud-inpainting.py', 'r').read())

##########################################################

#######
def create_video_data(BASE_PATH_RGB=None, BASE_PATH_DEPTH=None, BASE_TARGET_PATH=None, BATCH_SIZE=1):
	CLASS = ''
	## Set corruptions to generate and save images
	corruptions_to_generate = ['bit_error', 'h265_abr', 'h265_crf']

	# ## Create folders
	for corruption in corruptions_to_generate:
		if corruption == 'bit_error':
			TARGET_PATH = os.path.join(BASE_TARGET_PATH, corruption)
			if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)
		elif corruption == 'h265_abr': 
			TARGET_PATH = os.path.join(BASE_TARGET_PATH, corruption)
			if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)
		if corruption == 'h265_crf': 
			TARGET_PATH = os.path.join(BASE_TARGET_PATH, corruption) # , CLASS)
			if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)

	RGB_PATH = os.path.join(BASE_PATH_RGB, CLASS)
	DEPTH_PATH = os.path.join(BASE_PATH_DEPTH, CLASS)
	rgb_and_depth_dataset = RGBAndDepthDataset(RGB_PATH, DEPTH_PATH)
	rgb_and_depth_loader = data.DataLoader(rgb_and_depth_dataset, batch_size = BATCH_SIZE, 
							shuffle = False, num_workers = 0, drop_last = False)

	for idx, group in enumerate(rgb_and_depth_loader):
		
		data_curr, paths = group
		rgb_batch, depth_batch = data_curr[:,:3,:,:], data_curr[:,3,:,:]
		paths = list(paths)
		break

	base_savedir = BASE_TARGET_PATH 

	# ##########################################################

	ALL_CLASSES = [CLASS]

	# Save corruptions for all classes
	for i, CLASS in enumerate(ALL_CLASSES):
		RGB_PATH = os.path.join(BASE_PATH_RGB, CLASS)
		DEPTH_PATH = os.path.join(BASE_PATH_DEPTH, CLASS)
		rgb_and_depth_dataset = RGBAndDepthDataset(RGB_PATH, DEPTH_PATH)
		rgb_and_depth_loader = data.DataLoader(rgb_and_depth_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0, drop_last = False)
		
		for idx, group in enumerate(rgb_and_depth_loader):
			data_curr, paths = group
			rgb_batch, depth_batch = data_curr[:,:3,:,:], data_curr[:,3,:,:]
			paths = list(paths)
			
			all_rgb_files, all_depth_files = rgb_batch, depth_batch
			all_idx = paths

			for iii in tqdm.tqdm(range(len(all_rgb_files))):
				npyImage = all_rgb_files[iii].permute(1,2,0)*255
				npyImage = npyImage.numpy()
				npyImage= np.uint8(npyImage)
				depth_loaded = all_depth_files[iii].squeeze()#*65535

				idx_curr = paths[iii]

				motion_blur_vid_path = f"{base_savedir}/motion_blur_vid/{CLASS}/{idx_curr[:-5]}.mp4"
				zoom_blur_vid_path = f"{base_savedir}/zoom_blur_vid/{CLASS}/{idx_curr[:-5]}.mp4"

				h265_crf_sev4_path = f"{base_savedir}/h265_crf/{CLASS}/{idx_curr}"
				h265_abr_sev3_path = f"{base_savedir}/h265_abr/{CLASS}/{idx_curr}"
				bit_error_sev5_path = f"{base_savedir}/bit_error/{CLASS}/{idx_curr}"

				# Apply video distortions

				if bool(random.getrandbits(1)): #randomly pick either zoom or motion video for codec
					src_vid = zoom_blur_vid_path
				else:
					src_vid = motion_blur_vid_path
				dst_vid = './motion_video/tmp/out_vid.mp4'
				bit_error_frame_5 = bit_error(src_vid,dst_vid, 5, bit_error_sev5_path)
				h265_crf_frame_4 = h265_crf(src_vid,dst_vid, 4, h265_crf_sev4_path)
				h265_abr_frame_3 = h265_abr(src_vid,dst_vid, 3, h265_abr_sev3_path)



			
