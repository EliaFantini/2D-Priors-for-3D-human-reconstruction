import argparse
from create_dof import create_dof_data
import os

from motion_video.create_video import create_video_data
from create_non3d import create_non3d_data
from create_fog import create_fog_data
from motion_video.create_motion import create_motion_data

from zoom_blur_2d import zoom_blur_2d
from scale_pitch_roll_jitter import apply_pitch, apply_roll, apply_scale

parser = argparse.ArgumentParser(description='create 3dcc data')

parser.add_argument('--data_path', dest='data_path', help="path to clean rgb images")
parser.set_defaults(path_rgb='NONE')

parser.add_argument('--path_target', dest='path_target', help="path to store 3dcc generated images")
parser.set_defaults(path_target='NONE')

parser.add_argument('--batch_size', dest='batch_size', help="batch Size for processing the data", type=int)
parser.set_defaults(batch_size=1)

args = parser.parse_args()
data_path = args.data_path
path_target = args.path_target
batch_size = args.batch_size

# create a list of all the folders in the path_rgb
# create a list of all the folders in the path_depth

render_path = os.path.join(data_path, "render")
depth_path = os.path.join(data_path, "depth_F")

print("Fog 3D")
create_fog_data(BASE_PATH_RGB=render_path, BASE_PATH_DEPTH=depth_path, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)
print("XY-Motion Blur, Z-Motion Blur")
create_motion_data(BASE_PATH_RGB=render_path, BASE_PATH_DEPTH=depth_path, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)

print("Zoom Blur 2D")
zoom_blur_2d(BASE_PATH_RGB=render_path, BASE_TARGET_PATH=path_target)
print("Pitch, Roll")
#apply_pitch(BASE_PATH_RGB=render_path, BASE_TARGET_PATH=path_target)
apply_roll(BASE_PATH_RGB=render_path, BASE_TARGET_PATH=path_target)
print("Scale")
#apply_scale(BASE_PATH_RGB=render_path, BASE_TARGET_PATH=path_target)
print("Near Focus, Far Focus")
create_dof_data(BASE_PATH_RGB=render_path, BASE_PATH_DEPTH=depth_path, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)

print("CRF Compression, ABR Compression, Bit Error")
create_video_data(BASE_PATH_RGB=render_path, BASE_PATH_DEPTH=depth_path, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)
print("Low-light Noise, Color Quantization, ISO Noise")
create_non3d_data(BASE_PATH_RGB=render_path, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)
    

