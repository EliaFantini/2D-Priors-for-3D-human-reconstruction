"""
  Name: io_utils.py
  Author: Sasha Sax, CVGL
  Desc: Contains utilities for saving and loading information

  Usage: for import only
"""

from load_settings import settings

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix, Quaternion, Euler
    import utils
    from utils import create_camera, axis_and_positive_to_cube_face, cube_face_idx_to_skybox_img_idx
except:
    print("Can't import Blender-dependent libraries in io_utils.py...")


import ast
import csv
import glob
from itertools import groupby
import json
import logging
import math
from natsort import natsorted, ns
import numpy as np
import os
import sys

axis_and_positive_to_skybox_idx = {
    ("X", True): 1,
    ("X", False): 3,
    ("Y", True): 0,
    ("Y", False): 5,
    ("Z", True): 2,
    ("Z", False): 4
}

skybox_number_to_axis_and_rotation = {5: ('X', -math.pi / 2),
                                      0: ('X', math.pi / 2),
                                      4: ('Y', 0.0),
                                      3: ('Y', math.pi / 2),
                                      2: ('Y', math.pi),
                                      1: ('Y', -math.pi / 2)}

img_format_to_ext = {"png": 'png', "jpeg": "jpg", "jpg": "jpg"}
logger = settings.LOGGER


def delete_materials():
    ''' Deletes all materials in the scene. This can be useful for stanardizing meshes. '''
    # https://blender.stackexchange.com/questions/27190/quick-way-to-remove-thousands-of-materials-from-an-object
    C = bpy.context
    obj = C.object
    obj.data.materials.clear()


def get_file_name_for(dir, point_uuid, view_number, camera_uuid, task, ext):
    """
      Returns the filename for the given point, view, and task

      Args:
        dir: The parent directory for the model
        task: A string definint the task name
        point_uuid: The point identifier
        view_number: This is the nth view of the point
        camera_uuid: An identifier for the camera
        ext: The file extension to use
    """
    view_specifier = view_number
    filename = "point_{0}_view_{1}_domain_{2}.{3}".format(point_uuid, view_specifier, task, ext)
    return os.path.join(dir, filename)


def get_model_file(dir, typ='RAW'):
    if typ == 'RAW':
        model_file = settings.MODEL_FILE
    elif typ == 'SEMANTIC':
        model_file = settings.SEMANTIC_MODEL_FILE
    elif typ == 'RGB':
        model_file = settings.RGB_MODEL_FILE
    else:
        raise ValueError('Unknown type of model file: {0}'.format(typ))
        
    return os.path.join(dir, model_file)


def get_number_imgs(point_infos):
    n_imgs = 0
    n_imgs += sum([len(pi) for pi in point_infos])
       
    return n_imgs


def import_mesh(dir, typ='RAW'):
    ''' Imports a mesh with the appropriate processing beforehand.
    Args:
      dir: The dir from which to import the mesh. The actual filename is given from settings.
      typ: The type of mesh to import. Must be one of ['RAW', 'SEMANTIC', 'SEMANTIC_PRETTY', 'LEGO']
        Importing a raw model will remove all materials and textures.
    Returns:
      mesh: The imported mesh.
    '''
    model_fpath = get_model_file(dir, typ=typ)

    if '.obj' in model_fpath:
        bpy.ops.import_scene.obj(
            filepath=model_fpath,
            axis_forward=settings.OBJ_AXIS_FORWARD,
            axis_up=settings.OBJ_AXIS_UP)
        model = join_meshes()  # OBJs often come in many many pieces
        bpy.context.scene.objects.active = model
        
        if typ == 'RGB':
            return

        for img in bpy.data.images:  # remove all images
            bpy.data.images.remove(img, do_unlink=True)

        delete_materials()

    elif '.ply' in model_fpath:
        bpy.ops.import_mesh.ply(filepath=model_fpath)

    model = bpy.context.object

    return model


def join_meshes():
    ''' Takes all meshes in the scene and joins them into a single mesh.
    Args:
        None
    Returns:
        mesh: The single, combined, mesh 
    '''
    # https://blender.stackexchange.com/questions/13986/how-to-join-objects-with-python
    scene = bpy.context.scene
    obs = []
    for ob in scene.objects:
        if ob.type == 'MESH':
            obs.append(ob)

    ctx = bpy.context.copy()
    # one of the objects to join
    ctx["object"] = obs[0]
    ctx['active_object'] = obs[0]
    ctx['selected_objects'] = obs
    ctx["selected_editable_objects"] = obs
    # we need the scene bases as well for joining
    ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]
    bpy.ops.object.join(ctx)

    for ob in bpy.context.scene.objects:
        # whatever objects you want to join...
        if ob.type == 'MESH':
            return ob

def load_saved_points_of_interest(model_dir):
    """
      Loads all the generated points that have multiple views.

      Args:
        dir: Parent directory of the model. E.g. '/path/to/model/u8isYTAK3yP'

      Returns:
        point_infos: A list where each element is the parsed json file for a point
    """
    point_files = natsorted(glob.glob(os.path.join(model_dir, "point_info", "point_*.json")))

    point_infos = []
    for point_num, files_for_point in groupby(point_files, key=lambda x: parse_filename(x)['point_uuid']):
        pi = []
        print("point num : ", point_num)
        for view_file in files_for_point:
            with open(view_file) as f:
                pi.append(json.load(f))
        point_infos.append(pi)
    logger.info("Loaded {0} points of interest.".format(len(point_infos)))
    return point_infos



def parse_filename(filename):
    fname = os.path.basename(filename).split(".")[0]
    toks = fname.split('_')
    if toks[0] == "camera":
        point_uuid = toks[1]
        domain_name = toks[-1]
        view_num = toks[5]
    elif len(toks) == 6:
        point, point_uuid, view, view_num, domain, domain_name = toks
    elif len(toks) == 7:
        point, point_uuid, view, view_num, domain, domain_name, _ = toks

    return {'point_uuid': point_uuid, 'view_number': view_num, 'domain': domain_name}



if __name__ == '__main__':
    import argparse

    args = argparse.Namespace()
    load_settings(args)



  
