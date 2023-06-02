import os
import random

CLEAR_DATASET_PATH = "/mnt/c/Users/Elia/Desktop/input"
CORRUPTON_CODE_FOLDER_PATH= "/home/trial/3DCorruption/3DCommonCorruptions/create_3dcc"



# from CLEAR_DATASET_PATH get only NUM_OF_MODELS folders at random among all folders and from each of them get only 1 image at random
# and copy them in a tmp folder inside CLEAR_DATASET_PATH called tmp
os.system('cd ' + CLEAR_DATASET_PATH)

# get a list of all the folders in CLEAR_DATASET_PATH
list_of_folders_render = os.listdir(CLEAR_DATASET_PATH + "/input")
list_of_folders_depth = os.listdir(CLEAR_DATASET_PATH + "/depth")
# from each of them get only 1 image at random
imgs_paths = []
depth_paths = []
for folder in list_of_folders_render:
    rnd_img = random.choice(os.listdir(os.path.join(CLEAR_DATASET_PATH, "input",folder)))
    imgs_paths.append(os.path.join(CLEAR_DATASET_PATH, "input", folder, rnd_img))
for folder in list_of_folders_depth:
    rnd_img = random.choice(os.listdir(os.path.join(CLEAR_DATASET_PATH, "depth", folder)))
    depth_paths.append(os.path.join(CLEAR_DATASET_PATH, "depth", folder, rnd_img))

# copy them in a tmp folder inside CLEAR_DATASET_PATH called tmp
os.system('cd ' + CLEAR_DATASET_PATH)
# create the tmp folder if it does not exist
if not (os.path.isdir(os.path.join(CLEAR_DATASET_PATH, "tmp"))): os.makedirs(os.path.join(CLEAR_DATASET_PATH, "tmp"))
if not (os.path.isdir(os.path.join(CLEAR_DATASET_PATH, "tmp","render"))): os.makedirs(os.path.join(CLEAR_DATASET_PATH, "tmp","render"))
if not (os.path.isdir(os.path.join(CLEAR_DATASET_PATH, "tmp","depth_F"))): os.makedirs(os.path.join(CLEAR_DATASET_PATH, "tmp","depth_F"))
for img_path in imgs_paths:
    # copy the img_path in the tmp folder by renaming the image as {folder}_{rnd_img}_rgb.png
    os.system('cp ' + img_path + ' ' + os.path.join(CLEAR_DATASET_PATH, "tmp","render") + '/' + img_path.split('/')[-3] + '_' + img_path.split('/')[-2] + '_' + img_path.split('/')[-1])
for depth_path in depth_paths:
    os.system('cp ' + depth_path + ' ' + os.path.join(CLEAR_DATASET_PATH, "tmp", "depth_F")+ '/' + depth_path.split('/')[-3] + '_' + depth_path.split('/')[-2]+ '_' + depth_path.split('/')[-1])

os.system('cd ' + CORRUPTON_CODE_FOLDER_PATH)

# apply the corruption code to the tmp folder
os.system('python run.py --data_path ' + os.path.join(CLEAR_DATASET_PATH, "tmp") + ' --path_target ' + os.path.join(CLEAR_DATASET_PATH, "tmp"))

# create a subfolder of tmp called final
if not (os.path.isdir(os.path.join(CLEAR_DATASET_PATH, "tmp","final"))): os.makedirs(os.path.join(CLEAR_DATASET_PATH, "tmp","final"))
# starting from the tmp folder, scan every subfolder finding all files contained in them and copy them in the final folder, renaming them with the name of the original path to that file replacing / with _
# for example, if a file is in /tmp/render/1_1_1_rgb.png, it will be copied in /tmp/final/ as render_1_1_1_rgb.png, if a file is in /tmp/depth_F/5/1_1_1_rgb.png, it will be copied in /tmp/final/ as depth_F_5_1_1_1_rgb.png
for root, dirs, files in os.walk(os.path.join(CLEAR_DATASET_PATH, "tmp")):
    for file in files:
        # if the substring ["final","zoom_blur_vid","depth_F"] are contained in the root string, skip this file
        if "final" in root or "zoom_blur_vid" in root or "depth_F" in root: continue

        subpath = root.split("tmp/")[1]
        os.system('cp ' + os.path.join(root, file) + ' ' + os.path.join(CLEAR_DATASET_PATH, "tmp","final") + '/' + subpath.replace('/','_') + '_' + file.replace('/','_'))






