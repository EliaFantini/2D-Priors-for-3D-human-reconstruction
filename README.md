<p align="center">
  <img alt="🧍2D_Priors_for_3D_Human_Reconstruction " src="https://github.com/EliaFantini/2D-Priors-for-3D-human-reconstruction/assets/62103572/cbf0036b-bfc1-4fa8-a8fa-dd10c808f1a9">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/EliaFantini/2D-Priors-for-3D-human-reconstruction">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/EliaFantini/2D-Priors-for-3D-human-reconstruction">
  <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/EliaFantini/2D-Priors-for-3D-human-reconstruction">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/EliaFantini/2D-Priors-for-3D-human-reconstruction">
  <img alt="GitHub follow" src="https://img.shields.io/github/followers/EliaFantini?label=Follow">
  <img alt="GitHub fork" src="https://img.shields.io/github/forks/EliaFantini/2D-Priors-for-3D-human-reconstruction?label=Fork">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/EliaFantini/2D-Priors-for-3D-human-reconstruction?abel=Watch">
  <img alt="GitHub star" src="https://img.shields.io/github/stars/EliaFantini/2D-Priors-for-3D-human-reconstruction?style=social">
</p>

This repository contains code for the experiments of our project "Potential of 2D Priors for Improving Robustness of Ill-Posed 3D Reconstruction" for EPFL CS 503 Visual Intelligence. 

In this project we focus on the problem of comprehensive
human full-body mesh reconstruction from a single image,
which is assumed to be taken in daily settings. We tackle the problem of lack of information and corruption by integrating different 2D priors into the workflow to
enhance the robustness of 3D generative models. 

In other words, we explore the possibility to enhance 3D generative performance by leveraging pretrained 2D priors, and investigate different integration techniques for best performance. 

<img width="855" alt="image" src="https://github.com/EliaFantini/2D-Priors-for-3D-human-reconstruction/assets/62103572/ffe6593c-7c71-4c58-bb78-19472dc8c2ea">




<img width="736" alt="image" src="https://github.com/EliaFantini/2D-Priors-for-3D-human-reconstruction/assets/62103572/ec742316-5b99-44c3-9255-964bb8c962db">



## Quick presentation

![ezgif com-gif-maker](https://github.com/EliaFantini/2D-Priors-for-3D-human-reconstruction/assets/62103572/445d6996-0e77-447d-a7dc-9c4e2ccedced)

For more detailed info please read the [report](https://github.com/EliaFantini/2D-Priors-for-3D-human-reconstruction/blob/main/report.pdf).


## Authors 

-  [Fantini Elia](https://github.com/EliaFantini)
-  [Chengkun Li](https://github.com/CharlieLeee)
-  Ziwei Liu


## Contents
- [Potential of 2D Priors for Improving Robustness of Ill-Posed 3D Reconstruction](#potential-of-2d-priors-for-improving-robustness-of-ill-posed-3d-reconstruction)
- [Contents](#contents)
  - [0. Work done](#0-work-done)
  - [1. Environment Configuration](#1-environment-configuration)
  - [Training and testing](#training-and-testing)
  - [Code](#code)
  - [Data Augmentation](#data-augmentation)
- [Acknowledgements](#acknowledgements)

## 

### 0. Work done 

* Environment Configuration 
- [x] For rendering `THuman2.0` data with [JIFF](https://github.com/yukangcao/JIFF) code 
- [x] For training [PIFu](https://github.com/shunsukesaito/PIFu)
- [x] For data augmentation through commom corruptions 
* Data Preprocessing 
- [x] `THuman2.0` data rendering 
- [x] Data augmentation through commom corruptions 
* Model Training and testing 
- [x] Testing PIFu on augmented data 
- [x] Test-time adaptation of PIFu on the augmented data 
- [x] Post-training refinement with CLIP semantic loss
- [x] Naive CLIP feature fusion with `transform_concat` and `elementwise_additiion`
- [x] Multimodal learning with Domain Experts (we now support CLIP, DPT) fusion
### 1. Environment Configuration 

To setup the python environment used in this project, please follow the following steps:
```
conda create -n 2d3d python=3.8
conda activate 2d3d 

conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pillow
conda install scikit-image
conda install tqdm

pip install numpy cython
pip install menpo
pip install opencv-python



pip install pyembree # or alternatively use conda install -c conda-forge pyembree
pip install PyOpenGL



pip install ffmpeg

pip install rembg[gpu]
```
### Dataset Preparation
In this project, we rendered Thuman2.0 to the same format as RenderPeople (used in the original PIFu paper/code). To download our pre-rendered dataset, please `pip install gdown` and use this command:
```bash
gdown https://drive.google.com/u/2/uc?id=1py5ru62Rn6wpOX2LsmAthFs_PnQMeDfK
```
After downloading it, please extract to a location, in the following part of this README, we will refer to this location as `<dataroot>`, an example of <dataroot> can be `/scratch/izar/ckli/rendered_jiff_complete/`.
  
Some experiments will require some data to be arranged in a different way than training. For simplicity, we already provide them in the following link. Download it in the same way with command:
```bash
gdown https://drive.google.com/u/2/uc?id=1BiHPE9ptX0X5WMkyaqRAAGSm5npSVdTX
```
It will contain the folders `tta_clip`,`eval`, and `eval_masking`. The first one is required to run test on Self Supervised Finetuning, the second one to run evaluation on any kind kind of model we tested, the third one is required to ran the masking techniques comparison, as it contains corrupted masks.

### Training Commands

#### Training Vanilla PIFu with THuman2.0

> Notice: we use `wandb` for logging, if you don't use wandb, please comment the code related to wandb in `train_shape.py`.

```bash
python -m apps.train_shape --dataroot <dataroot>  --checkpoints_path ./baseline --batch_size 16  --num_epoch 5 --name vanilla-baseline
```

#### Test Time Adaptation with Corruption data
 
Test time adaptation on vanilla PIFu aims to use the corrupted data to enhance the model's robustness. 

Finetuning for 5 epochs with 5 samples: 
  
```
python -m apps.train_tta --dataroot <dataroot>  --checkpoints_path ./tta_baseline_5_sample --load_netG_checkpoint_path <path_to_vanilla_baseline_netG>  --batch_size 16 --tta_adapt_sample 5 --num_epoch 5
```


### Training the baseline geometric network

Training the baseline geometric network using the augmented data. 

Training for 10 epochs: 
```
python -m apps.train_shape --dataroot <path/to/the/dataset>  --checkpoints_path ./baseline_G --batch_size 16   --num_epoch 10 --phase dataaug
```
whenever you want to continue from last checkpoint use `--continue_train` paired with `--resume_epoch` arguments.

### Naive CLIP multimodal learning

Use `tf_concat` or `add` for feature_fusion option.
```
python -m apps.train_shape --dataroot <data_root> --checkpoints_path ./baseline_clip_G --batch_size 16 --num_epoch 5 --name clip_baseline_correct --feature_fusion add --learning_rate 0.001 --use_clip_encoder True
```
  
### Training the model with Experts Fusion module
Training the model with multimodel learning with DPT (depth prior) and CLIP prior. 

Training for 5 epochs: 

```
python -m apps.train_shape --dataroot <path/to/the/dataset>  --checkpoints_path ./primser --batch_size 16 --mlp_dim 258 1024 512 256 128 1 --use_dpt True --freeze_encoder True --feature_fusion prismer --load_netG_checkpoint_path <path/to/PIFu/net_G> --num_epoch 5 --name prismer --use_clip_encoder True
```

Continuing training: 
```
CUDA_VISIBLE_DEVICES=0 python -m apps.train_shape --dataroot <path/to/the/dataset>  --checkpoints_path ./primser --batch_size 16 --mlp_dim 258 1024 512 256 128 1 --use_dpt True --freeze_encoder True --feature_fusion prismer --load_netG_checkpoint_path <path/to/PIFu/net_G> --num_epoch 5 --name prismer --use_clip_encoder True --continue_train --checkpoints_path <path/to/PIFu/primser> --resume_epoch 3
```
#### Additional ablation experiments
CLIP + image
```
python -m apps.train_shape --dataroot /scratch/izar/ckli/rendered_jiff_complete/ --checkpoints_path ./primser --batch_size 16 --mlp_dim 258 1024 512 256 128 1 --freeze_encoder True --feature_fusion prismer --load_netG_checkpoint_path /scratch/izar/ckli/PIFu/checkpoints/net_G --num_epoch 5 --name prismer_no_dpt --use_clip_encoder True
```
Continue CLIP+Image
```
python -m apps.train_shape --dataroot /scratch/izar/ckli/rendered_jiff_complete/ --checkpoints_path ./primser --batch_size 16 --mlp_dim 258 1024 512 256 128 1 --freeze_encoder True --feature_fusion prismer --load_netG_checkpoint_path /scratch/izar/ckli/PIFu/checkpoints/net_G --num_epoch 5 --name prismer_no_dpt --use_clip_encoder True --continue_train
``` 
Only CLIP
```
python -m apps.train_shape --dataroot /scratch/izar/ckli/rendered_jiff_complete/  --checkpoints_path ./primser --batch_size 16 --mlp_dim 258 1024 512 256 128 1  --freeze_encoder True --feature_fusion prismer --load_netG_checkpoint_path /scratch/izar/ckli/PIFu/checkpoints/net_G --num_epoch 5 --name prismer_clip_only --use_clip_encoder True --prismer_only_clip True --resume_epoch 3
```
  
#### Running Self-Supervised Refinement

This experiment will use the downloaded `tta_clip` data mentioned in Dataset Preparation section. Copy the path to that folder in the variable DATA_PATH inside `apps/clip_loss_tta.py`. Results can be saved in the preferred location by changing the `results_path` variable in the same file, or by default a `results` folder will be created inside DATA_PATH. Set AVERAGE variable to True to calculate averaged Clip embedding instead of applying Clip Loss in a sequential manner.
  Then run the following command from project's root folder:
  
```
sh scripts/test_clip.sh
```
#### Experiment on different masking production techniques and evaluation of their performance
This experiment will use the downloaded `eval_masking` data mentioned in Dataset Preparation section. Copy the path to that folder in the variable DATA_PATH inside `apps/eval_masking.py`. Results can be saved in the preferred location by changing the `results_path` variable in the same file.
  Then run the following command from project's root folder:
  
```
sh scripts/test_masking.sh
```
  

#### Running evaluation
This experiment will use the downloaded `eval` data mentioned in Dataset Preparation section. Copy the path to that folder in the variable DATA_PATH inside `apps/clip_loss_tta.py`. Results can be saved in the preferred location by changing the `results_path` variable in the same file. To evaluate a specific model, load its checkpoint by replacing the path to it in the variable `CHECKPOINTS_NETG_PATH`  inside `scripts/test.sh`
  Then run the following command from project's root folder:
  
```
sh scripts/test.sh
```
  

### Code 
This codebase provides code for: 

- **Models, data and rendering**
  
Our backbone is PIFu model. We included the vanilla PIFu model, PIFu variants, PIFu with CLIP loss, and other helper functions in the folder `lib/model`. The train and evaluation dataset fed into the network are processed by the code in `lib/data`. 

During the training, we need to render images of the specified angles from the human mesh models, and the code for rendering are in the folder `lib/renderer`. 

Other files under the `lib` directory are also helper functions for the training. 

- **Data augmentation via corruption**

We augment the dataset by corrupting part of the rendered images, and the code for creating the dataset is in the folder `data/3d_common_corruptions/create_3dcc/motions_video`. 

- **Training and evaluating**
  
The code for training and evaluating, and also the camera settings are in the folder `apps`. 
  
- **Experiments**
  
 The `experiments` folder contains some code we used to generate plots and tables on the .npy and pickles generated from our experiments, as well as some code to conduct some very specific experiments.

### Data Augmentation 

In this part of the project, we augment the dataset with corrupted images. We borrowed part of the code from the paper: [**3D Common Corruptions and Data Augmentation**](https://3dcommoncorruptions.epfl.ch/). 

We utilize 4 different kinds of corruptions, including low_light, iso_noise, camera_roll, and zoom_blur. Specifically, for each model, we have 360 rendered images from 360 different angles. For each model's rendered images, we pick `1/5` of data for each kind of corruption and leave the rest `1/5` as they are. 

To corrupted the images, use the command line
```
python -m data.3d_common_corruptions.create_3dcc.corrupt.py --RENDER_PATH <path/to/the/rendered/images> --MASK_PATH <path/to/the/masks>
```

## Acknowledgements 

Our implementation is based on [PIFu](https://github.com/shunsukesaito/PIFu), [JIFF](https://github.com/yukangcao/JIFF), [3D Common Corruptions](https://3dcommoncorruptions.epfl.ch/), [CLIP](https://github.com/openai/CLIP), and [Prismer](https://shikun.io/projects/prismer). 

## 🛠 Skills
Python, Pytorch, 2D Priors, CLIP, CLIP Loss, Prismer. 3D human full-body mesh reconstruction from single images,
robustness to real-life corruption analysis, self-supervised refinement with CLIP Loss, multimodal learning.

## 🔗 Links
[Chengkun Li portfolio](https://charlieleee.github.io/)


[![My portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://eliafantini.github.io/Portfolio/)
[![My linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)

