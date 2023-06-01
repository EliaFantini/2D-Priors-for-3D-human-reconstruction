"""
  Name: create_images_utils.py
  Desc: Contains utilities which can be used to run 
  
"""

import logging
import os
import sys
from load_settings import settings

import pdb 
try:
    import bpy
    import numpy as np
    from mathutils import Vector, Matrix, Quaternion, Euler
    import io_utils
    from io_utils import get_number_imgs
    import utils
    from utils import Profiler

except:
    print("Can't import Blender-dependent libraries in io_utils.py. Proceeding, and assuming this is kosher...")

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def start_logging():
    ''' '''
    #   global logger
    logger = io_utils.create_logger(__name__)
    utils.set_random_seed()
    basepath = os.getcwd()
    return logger, basepath


def setup_and_render_image(task_name, object_class, basepath, view_number, view_dict, execute_render_fn, logger=None,
                           clean_up=True, face_to_object=None):
    ''' Mutates the given camera and uses it to render the image called for in 
        'view_dict'
    Args:
        task_name: task name + subdirectory to save images
        basepath: model directory
        view_number: The index of the current view
        view_dict: A 'view_dict' for a point/view
        execute_render_fn: A function which renders the desired image
        logger: A logger to write information out to
        clean_up: Whether to delete cameras after use
    Returns:
        None (Renders image)
    '''
    scene = bpy.context.scene
    camera_uuid = view_dict["camera_uuid"]
    point_uuid = view_dict["point_uuid"]
    if "camera_rotation_original" not in view_dict:
        view_dict["camera_rotation_original"] = view_dict["camera_original_rotation"]
    
    camera, camera_data, scene = utils.get_or_create_camera(
        location=view_dict['camera_location'],
        rotation=view_dict["camera_rotation_original"],
        field_of_view=view_dict["field_of_view_rads"],
        scene=scene,
        camera_name='RENDER_CAMERA')

    save_path = io_utils.get_file_name_for(
        dir=get_save_dir(basepath, object_class),
        point_uuid=point_uuid,
        view_number=view_number,
        camera_uuid=camera_uuid,
        task=task_name,
        ext=io_utils.img_format_to_ext[settings.PREFERRED_IMG_EXT.lower()])
    # Aim camera at target by rotating a known amount
    camera.rotation_euler = Euler(view_dict["camera_rotation_original"])
    camera.rotation_euler.rotate(
        Euler(view_dict["camera_rotation_from_original_to_final"]))
    
    execute_render_fn(scene, save_path, face_to_object)

    if clean_up:
        utils.delete_objects_starting_with("RENDER_CAMERA")  # Clean up

    
def get_save_dir(basepath, task_name):
    if settings.CREATE_PANOS:
        return os.path.join(basepath, 'pano', task_name)
    else:
        return os.path.join(basepath, task_name)
