import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

# Data path 
RENDER_PATH = "/scratch/izar/ziwliu/complete_jiff_corrupted/RENDER"
MASK_PATH = "/scratch/izar/ziwliu/complete_jiff_corrupted/MASK"
CLEAN_MASK_PATH = "/scratch/izar/ziwliu/rendered_jiff_complete/MASK"

# Parameters 
NUM_OF_EACH_TYPE = 72
batch_size = 1 

def convert_grey_to_white(input_path, output_path):
    # Open the image file
    image = Image.open(input_path)

    # Convert the image to RGB mode if it's in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Get the pixel data from the image
    pixels = image.load()

    # Iterate over each pixel and convert grey to white
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = pixels[x, y]
            # Check if the pixel is grey (r = g = b)
            if r == g == b and r != 0:
                # Convert the grey pixel to white (255, 255, 255)
                pixels[x, y] = (255, 255, 255)
    # Save the modified image
    image.save(output_path)

def create_path_mask(corr_mask_path, clean_mask_path, file_name):
    file_name_parts = file_name.split("_")
    angle = "_".join(file_name_parts[0:3])
    corr_mask = os.path.join(corr_mask_path, file_name)
    clean_mask = os.path.join(clean_mask_path, angle + '.png')
    return corr_mask, clean_mask

def process_corrupted_mask(corr_mask_path, clean_mask_path, file_name):
    file_name_parts = file_name.split("_")
    type = "_".join(file_name_parts[3:]).split(".")[0]
    corr_mask, clean_mask = create_path_mask(corr_mask_path, clean_mask_path, file_name)
    if (type == 'low_light') or (type == 'iso_noise'):
        # print(f"The mask is of the corruption {type}")
        # print(f'rm {corr_mask}')
        # print(f"cp {clean_mask} {corr_mask}")
        os.system('rm ' + corr_mask)
        os.system('cp ' + clean_mask + ' ' + corr_mask)
        
    elif type == 'zoom_blur':
        convert_grey_to_white(corr_mask, corr_mask)
        

object_list = [str(num).zfill(4) for num in range(101)]
object_list.remove('0000')
for object in object_list: 
    print(f"Object is {object}")
    object_path_mask = os.path.join(MASK_PATH, object)
    object_path_mask_clean = os.path.join(CLEAN_MASK_PATH, object)
    list_of_masks = os.listdir(object_path_mask)
    for file_name in list_of_masks:
        process_corrupted_mask(object_path_mask, object_path_mask_clean, file_name)

sys.exit("Processing of masks finished")

