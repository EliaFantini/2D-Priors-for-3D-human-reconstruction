import sys
import os

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import natsort



class RGBAndDepthDataset(Dataset):
    
    """
        Dataset wrapper to load clean RGBs and corresponding depth maps (euclidean)
        for a given folder.
        
        **IMPORTANT: RGB image and depth map are returned as one tensor (stacked along 
                     the channel axis).
        
        Attributes
        ----------
        rgb_path : str
            path to RGB images
        depth_path : str
            path to depth maps

        total_rgb_imgs : List(str)
            natural-sorted list of filenames of all RGB images
        total_depth_imgs : List(str)
            natural-sorted list of filenames of all depth images

    """
    
    def __init__(self, rgb_path, depth_path, change_depth = False):
        
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.change_depth = change_depth

        # load file names for each data type
        rgb_imgs = os.listdir(rgb_path)
        depth_imgs = os.listdir(depth_path)
        
        # natural-sort the file names
        self.total_rgb_imgs = natsort.natsorted(rgb_imgs)
        self.total_depth_imgs = natsort.natsorted(depth_imgs)
    
    def __len__(self):
        return len(self.total_rgb_imgs)

    def __getitem__(self, idx):
        
        """
        assert(self.total_rgb_imgs[idx].split('domain')[0] == self.total_depth_imgs[idx].split('domain')[0],
                "viewpoints don't match")
        """
        
        image_loc = os.path.join(self.rgb_path, self.total_rgb_imgs[idx])
        depth_loc = os.path.join(self.depth_path, self.total_depth_imgs[idx])

        #use for taskonomy images
        image = TF.to_tensor(Image.open(image_loc).convert("RGB"))
        depth = TF.to_tensor(Image.open(depth_loc))
        if self.change_depth:
            depth_normalized = depth.float() / depth.max()*0.5
        else:
            depth_normalized = depth.float() / 8000.

        depth_normalized[depth_normalized >= 1.0] = depth_normalized[depth_normalized < 1.0].max()
        
        
        return (torch.cat((image, depth_normalized), dim = 0), self.total_rgb_imgs[idx])

"""
test_batch_images = None
test_batch_paths = None 

for idx, group in enumerate(rgb_and_depth_loader):
    image, paths = group
    test_batch_images = image
    test_batch_paths = list(paths)
    break
    
print(image.size())
print(paths)
print(group)
plt.imshow(image[2,0:3].permute(1,2,0))
"""

class RGBAndReshadeDataset(Dataset):
    
    """
        Dataset wrapper to load clean RGBs and corresponding depth maps (euclidean)
        for a given folder.
        
        **IMPORTANT: RGB image and depth map are returned as one tensor (stacked along 
                     the channel axis).
        
        Attributes
        ----------
        rgb_path : str
            path to RGB images
        depth_path : str
            path to depth maps

        total_rgb_imgs : List(str)
            natural-sorted list of filenames of all RGB images
        total_depth_imgs : List(str)
            natural-sorted list of filenames of all depth images

    """
    
    def __init__(self, rgb_path, reshade_path):
        
        self.rgb_path = rgb_path
        self.reshade_path = reshade_path
        
        # load file names for each data type
        rgb_imgs = os.listdir(rgb_path)
        reshade_imgs = os.listdir(reshade_path)
        
        # natural-sort the file names
        self.total_rgb_imgs = natsort.natsorted(rgb_imgs)
        self.total_reshade_imgs = natsort.natsorted(reshade_imgs)
    
    def __len__(self):
        return len(self.total_rgb_imgs)

    def __getitem__(self, idx):
        
        """
        assert(self.total_rgb_imgs[idx].split('domain')[0] == self.total_depth_imgs[idx].split('domain')[0],
                "viewpoints don't match")
        """
        
        image_loc = os.path.join(self.rgb_path, self.total_rgb_imgs[idx])
        reshade_loc = os.path.join(self.reshade_path, self.total_reshade_imgs[idx])

        #use for taskonomy images
        image = TF.to_tensor(Image.open(image_loc).convert("RGB"))
        reshade = TF.to_tensor(Image.open(reshade_loc))
        
        
        return (torch.cat((image, reshade), dim = 0), self.total_rgb_imgs[idx])



class RGBDataset(Dataset):
    
    """
        Dataset wrapper to load clean RGBs and corresponding depth maps (euclidean)
        for a given folder.
        
        **IMPORTANT: RGB image and depth map are returned as one tensor (stacked along 
                     the channel axis).
        
        Attributes
        ----------
        rgb_path : str
            path to RGB images
        depth_path : str
            path to depth maps

        total_rgb_imgs : List(str)
            natural-sorted list of filenames of all RGB images
        total_depth_imgs : List(str)
            natural-sorted list of filenames of all depth images

    """
    
    def __init__(self, rgb_path):
        
        self.rgb_path = rgb_path
        
        # load file names for each data type
        rgb_imgs = os.listdir(rgb_path)
        
        # natural-sort the file names
        self.total_rgb_imgs = natsort.natsorted(rgb_imgs)
    
    def __len__(self):
        return len(self.total_rgb_imgs)

    def __getitem__(self, idx):
        
        """
        assert(self.total_rgb_imgs[idx].split('domain')[0] == self.total_depth_imgs[idx].split('domain')[0],
                "viewpoints don't match")
        """
        
        image_loc = os.path.join(self.rgb_path, self.total_rgb_imgs[idx])

        #use for taskonomy
        image = TF.to_tensor(Image.open(image_loc).convert("RGB"))
        
        return (image, self.total_rgb_imgs[idx])