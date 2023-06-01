# Potential of 2D Priors for Improving Robustness of Ill-Posed 3D Reconstruction

This repository contains code for the experiments for the EPFL CS 503 Visual Intelligence project. 

## Table of Contents

- [Potential of 2D Priors for Improving Robustness of Ill-Posed 3D Reconstruction](#potential-of-2d-priors-for-improving-robustness-of-ill-posed-3d-reconstruction)
  - [Table of Contents](#table-of-contents)
  - [Training and Evaluating the models](#training-and-evaluating-the-models)
  - [Code](#code)
    - [Models, data and rendering](#models-data-and-rendering)
    - [Data augmentation via corruption](#data-augmentation-via-corruption)
    - [Training and evaluating](#training-and-evaluating)
  - [Procedure to build the environment needed to run the code](#procedure-to-build-the-environment-needed-to-run-the-code)
    - [For original PIFu](#for-original-pifu)
    - [For PIFuHD](#for-pifuhd)


## Training and Evaluating the models 

To run the code, use the following prompt: 

For training the baseline geometric network

`python -m apps.train_shape --dataroot <path/to/the/dataset>  --checkpoints_path ./baseline_G --batch_size 16  --gpu_ids "0,1" --num_epoch 10`

For continuning training shape network baseline

`CUDA_VISIBLE_DEVICES=0 python -m apps.train_shape --dataroot <path/to/the/dataset>  --checkpoints_path ./baseline_G --batch_size 16   --num_epoch 30 --pin_memory --continue_train --resume_epoch 9 --checkpoints_path ./baseline_G --name vanilla-baseline`

For training the baseline geometric network with CLIP
`python -m apps.train_shape --dataroot <path/to/the/dataset>  --checkpoints_path ./baseline_clip_G --batch_size 16  --num_epoch 5 --name clip_baseline --feature_fusion add --learning_rate 0.001 --use_clip_encoder True`

For continuing training shape network with CLIP 

`python -m apps.train_shape --dataroot <path/to/the/dataset>  --checkpoints_path ./baseline_clip_G --batch_size 16  --num_epoch 5 --name clip_baseline --feature_fusion add --learning_rate 0.001 --continue_train --resume_epoch 4`

## Code 
### Models, data and rendering
Our backbone is PIFu model. We included the vanilla PIFu model, PIFu variants, PIFu with CLIP loss, and other helper functions in the folder `lib/model`. The train and evaluation dataset fed into the network are processed by the code in `lib/data`. 

During the training, we need to render images of the specified angles from the human mesh models, and the code for rendering are in the folder `lib/renderer`. 

Other files under the `lib` directory are also helper functions for the training. 

### Data augmentation via corruption 

We augment the dataset by corrupting part of the rendered images, and the code for creating the dataset is in the folder `data/3d_common_corruptions/create_3dcc/motions_video`. 

### Training and evaluating
The code for training and evaluating, and also the camera settings are in the folder `apps`. 

## Procedure to build the environment needed to run the code

### For original PIFu 

```
conda create -n orig_pifu python=3.8
conda activate orig_pifu 

conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pillow
conda install scikit-image
conda install tqdm

pip install numpy cython
pip install menpo
pip install opencv-python

# in case of error with numpy
pip uninstall numpy
pip install numpy==1.22.4

# in case of error "cannot marching cubes"
# Open PIFu/lib/mesh_util.py and change line 45 to:
# verts, faces, normals, values = measure.marching_cubes(sdf, 0.5)
```

### For PIFuHD
```
pip install pyembree
# conda install -c conda-forge pyembree
pip install PyOpenGL
# freeglut (use sudo apt-get install freeglut3-dev for ubuntu users)
# this one we can't do it
pip install ffmpeg

pip install rembg[gpu]

pip uninstall numpy
pip install numpy==1.22.4
```