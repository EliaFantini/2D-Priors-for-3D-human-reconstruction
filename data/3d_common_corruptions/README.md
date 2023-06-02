# 3D Common Corruptions and Data Augmentation

This repository is the code that we use for data augmentation of corrupted data. We borrowed the code from the paper: [**3D Common Corruptions and Data Augmentation**](https://3dcommoncorruptions.epfl.ch/). 

Citation of the paper: 

```
@inproceedings{kar20223d,
  title={3D Common Corruptions and Data Augmentation},
  author={Kar, O{\u{g}}uzhan Fatih and Yeo, Teresa and Atanov, Andrei and Zamir, Amir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18963--18974},
  year={2022}
}
```

## Table of Contents

- [3D Common Corruptions and Data Augmentation](#3d-common-corruptions-and-data-augmentation)
  - [Table of Contents](#table-of-contents)
  - [Generate the corrupted images](#generate-the-corrupted-images)
  - [An overview of the corruptions](#an-overview-of-the-corruptions)
  - [Details on generating individual corruptions](#details-on-generating-individual-corruptions)

## Generate the corrupted images 
This code generates all corruptions mentioned in our report. Inside the code the variable CLEAR_DATASET_PATH has to be changed to the root folder of the training dataset

`python -m 3d_common_corruptions.create_3dcc.corrupt_v0.py
## Generate the corrupted images with corrupted masks

This code generates corruptions that mostly degradate final performance as mentioned in our report, and it also creates corrupted masks. 

`python -m 3d_common_corruptions.create_3dcc.corrupt.py --RENDER_PATH <path/to/the/rendered/images> --MASK_PATH <path/to/the/masks>`

## An overview of the corruptions

We utilize 4 different kinds of corruptions, including low_light, iso_noise, camera_roll, and zoom_blur. Specifically, for each model, we have 360 rendered images from 360 different angles. For each model's rendered images, we pick `1/5` of data for each kind of corruption and leave the rest `1/5` as they are. 

## Details on generating individual corruptions

It is also possible to generate corrupted images for individually selected corruption types. For this, you can browse the folder create_3dcc and use the corresponding function. For example, if you want to generate 3D Fog data, check the corresponding script `create_fog.py` which has the following function:

```bash
def create_fog_data(BASE_PATH_RGB=None, BASE_PATH_DEPTH=None, BASE_TARGET_PATH=None, BATCH_SIZE=1):
    ....
```
Then you can pass the required arguments to indicate location of your clean images and their corresponding depth, path to save images, and batch size. Similar process can be performed for other corruption types too, namely:
```bash
create_dof.py # Near/Far Focus
create_flash.py #Flash
create_fog.py #Fog 3D
create_multi_illumination.py #Multi-Illumination
create_shadow.py #Shadow
create_non3d.py # Low-light Noise, ISO Noise, Color Quantization
```
