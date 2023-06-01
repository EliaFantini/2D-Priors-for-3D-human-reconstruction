import os
import random
import sys
from zoom_blur_2d import zoom_blur_2d
from scale_pitch_roll_jitter import apply_roll
from create_non3d import create_non3d_data_iso_noise, create_non3d_data_low_light

# Code Path 
CORRUPTON_CODE_FOLDER_PATH = "/home/ziwliu/3DCommonCorruptions/create_3dcc"
ICON_REPO_FOLDER_PATH = "/home/ziwliu/ICON-master"

# Data path 
RENDER_PATH = "/scratch/izar/ziwliu/complete_jiff_corrupted/RENDER"
MASK_PATH = "/scratch/izar/ziwliu/complete_jiff_corrupted/MASK"

# Parameters 
NUM_OF_EACH_TYPE = 72
batch_size = 1 

def create_path(object_path_render, object_path_mask, angle, type):
    img_path_render = os.path.join(object_path_render, angle[0:-4] + '.jpg')
    img_path_mask = os.path.join(object_path_mask, angle[0:-4] + '.png')
    save_path_render = os.path.join(object_path_render, angle[0:-4] + '_' + type + '.jpg')
    save_path_mask = os.path.join(object_path_mask, angle[0:-4] + '_' + type + '.png')
    return img_path_render, img_path_mask, save_path_render, save_path_mask

def corrupt(list_of_angles_type, object_path_render, object_path_mask, type):
    if type == 'low_light':
        for angle in list_of_angles_type:
            img_path_render, img_path_mask, save_path_render, save_path_mask = create_path(object_path_render, object_path_mask, angle, type)
            create_non3d_data_low_light(BASE_PATH_RGB=img_path_render, BASE_TARGET_PATH=save_path_render, BATCH_SIZE=batch_size)
            create_non3d_data_low_light(BASE_PATH_RGB=img_path_mask, BASE_TARGET_PATH=save_path_mask, BATCH_SIZE=batch_size)
            os.system('rm ' + img_path_render)
            os.system('rm ' + img_path_mask)

    elif type == 'iso_noise':
        for angle in list_of_angles_type:
            img_path_render, img_path_mask, save_path_render, save_path_mask = create_path(object_path_render, object_path_mask, angle, type)
            create_non3d_data_iso_noise(BASE_PATH_RGB=img_path_render, BASE_TARGET_PATH=save_path_render, BATCH_SIZE=batch_size)
            create_non3d_data_iso_noise(BASE_PATH_RGB=img_path_mask, BASE_TARGET_PATH=save_path_mask, BATCH_SIZE=batch_size)
            os.system('rm ' + img_path_render)
            os.system('rm ' + img_path_mask)
        
    elif type == 'camera_roll':
        for angle in list_of_angles_type:
            img_path_render, img_path_mask, save_path_render, save_path_mask = create_path(object_path_render, object_path_mask, angle, type)
            apply_roll(BASE_PATH_RGB=img_path_render, BASE_TARGET_PATH=save_path_render)
            apply_roll(BASE_PATH_RGB=img_path_mask, BASE_TARGET_PATH=save_path_mask)
            os.system('rm ' + img_path_render)
            os.system('rm ' + img_path_mask)
        
    elif type == 'zoom_blur':
        for angle in list_of_angles_type:
            img_path_render, img_path_mask, save_path_render, save_path_mask = create_path(object_path_render, object_path_mask, angle, type)
            zoom_blur_2d(BASE_PATH_RGB=img_path_render, BASE_TARGET_PATH=save_path_render)
            zoom_blur_2d(BASE_PATH_RGB=img_path_mask, BASE_TARGET_PATH=save_path_mask)
            os.system('rm ' + img_path_render)
            os.system('rm ' + img_path_mask)
        
    elif type == 'original':
        for angle in list_of_angles_type:
            img_path_render, img_path_mask, save_path_render, save_path_mask = create_path(object_path_render, object_path_mask, angle, type)
            os.system('cp ' + img_path_render + ' ' + save_path_render)
            os.system('cp ' + img_path_mask + ' ' + save_path_mask)
            os.system('rm ' + img_path_render)
            os.system('rm ' + img_path_mask)

object_list = [str(num).zfill(4) for num in range(101, 121)]
for object in object_list: 
    print(f"Object is {object}")
    object_path_render = os.path.join(RENDER_PATH, object)
    object_path_mask = os.path.join(MASK_PATH, object)
    list_of_angles = os.listdir(object_path_render)
    # Pick out ones for different corruptions types. 
    random.shuffle(list_of_angles)
    list_of_angles_low_light = list_of_angles[0:NUM_OF_EACH_TYPE]
    list_of_angles_iso_noise = list_of_angles[NUM_OF_EACH_TYPE:2*NUM_OF_EACH_TYPE]
    list_of_angles_camera_roll = list_of_angles[2*NUM_OF_EACH_TYPE:3*NUM_OF_EACH_TYPE]
    list_of_angles_zoom_blur = list_of_angles[3*NUM_OF_EACH_TYPE:4*NUM_OF_EACH_TYPE]
    list_of_angles_original = list_of_angles[4*NUM_OF_EACH_TYPE:5*NUM_OF_EACH_TYPE]
    # Corrupt images and masks 
    corrupt(list_of_angles_low_light, object_path_render, object_path_mask, 'low_light')
    corrupt(list_of_angles_iso_noise, object_path_render, object_path_mask, 'iso_noise')
    corrupt(list_of_angles_camera_roll, object_path_render, object_path_mask, 'camera_roll')
    corrupt(list_of_angles_zoom_blur, object_path_render, object_path_mask, 'zoom_blur')
    corrupt(list_of_angles_original, object_path_render, object_path_mask, 'original')
    print((f"Corruption of object {object} finished"))

sys.exit("Corruption finished")

