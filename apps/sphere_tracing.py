# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from enum import IntEnum

from camera.camera import Camera
import torch
import torch.nn.functional as F

from tqdm import tqdm


# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


# -- Supersample / Ray jitter --

def generate_default_grid(width, height, device=None):
    h_coords = torch.arange(height, device=device)
    w_coords = torch.arange(width, device=device)
    return torch.meshgrid(h_coords, w_coords)  # return pixel_y, pixel_x






# -- Ray gen --

def _to_ndc_coords(pixel_x, pixel_y, width, height):
    pixel_x = 2 * (pixel_x / width) - 1.0
    pixel_y = 2 * (pixel_y / height) - 1.0
    return pixel_x, pixel_y

class CameraFOV(IntEnum):
    """Camera's field-of-view can be defined by either of the directions"""
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2    # Used by wide fov lens (i.e. fisheye)


def generate_pinhole_rays(width, height, device ):
    """Default ray generation function for pinhole cameras.

    This function assumes that the principal point (the pinhole location) is specified by a 
    displacement (camera.x0, camera.y0) in pixel coordinates from the center of the image. 

    The Kaolin camera class does not enforce a coordinate space for how the principal point is specified,
    so users will need to make sure that the correct principal point conventions are followed for 
    the cameras passed into this function.

    Args:
        camera (kaolin.render.camera): The camera class. 
        coords_grid (torch.FloatTensor): Grid of coordinates of shape [H, W, 2].

    Returns:
        (wisp.core.Rays): The generated pinhole rays for the camera.
    """

    pixel_y, pixel_x = generate_centered_pixel_coords(width, height, device=device)
    pixel_x = pixel_x.to(device)
    pixel_y = pixel_y.to(device)

    x0 = 0.0
    y0 = 0.0
    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - x0
    pixel_y = pixel_y + y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    pixel_x, pixel_y = _to_ndc_coords(pixel_x, pixel_y, width, height)

    ray_dir = torch.stack((pixel_x * tan_half_fov(CameraFOV.HORIZONTAL),
                           -pixel_y * tan_half_fov(CameraFOV.VERTICAL),
                           -torch.ones_like(pixel_x)), dim=-1)

    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)




    look_at = torch.zeros( (4, 3), dtype=torch.float32, device=device)
    camera_position = torch.tensor( [ [0, 0, 2],
                                                [2, 0, 0],
                                                [0, 0, -2],
                                                [-2, 0, 0]  ]  , dtype=torch.float32, device=device)
    camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=device).repeat(4, 1,)
    camera = Camera.from_args(eye=camera_position[0],
                                        at=look_at[0],
                                        up=camera_up_direction[0],
                                        fov_distance=1.0,
                                        width=512,
                                        height=512,
                                      dtype=torch.float32)
    
    ray_orig, ray_dir = camera.inv_transform_rays(ray_orig, ray_dir)

    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    ray_orig, ray_dir = ray_orig[0], ray_dir[0]  # Assume a single camera

    return ray_orig, ray_dir

def tan_half_fov(self, camera_fov_direction: CameraFOV = CameraFOV.VERTICAL) -> torch.FloatTensor:
        r"""tan(fov/2) in radians

        Args:
            camera_fov_direction (optional, CameraFOV):
                the leading direction of the fov. Default: vertical

        Returns:
            (torch.Tensor): tan(fov/2) in radians, of size :math:`(\text{num_cameras},)`
        """
        if camera_fov_direction is CameraFOV.HORIZONTAL:
            tanHalfAngle = torch.tensor((512 / 2.0) / 2)
        elif camera_fov_direction is CameraFOV.VERTICAL:
            tanHalfAngle = torch.tensor((512 / 2.0) / 3)
        else:
            raise ValueError(f'Unsupported CameraFOV direction enum given to tan_half_fov: {camera_fov_direction}')
        return tanHalfAngle




def inv_transform_rays(ray_orig: torch.Tensor, ray_dir: torch.Tensor):
        r"""Transforms rays from camera space to world space (hence: "inverse transform").

        Apply rigid transformation of the camera extrinsics.
        The camera coordinates are cast to the precision of the vectors argument.

        Args:
            ray_orig (torch.Tensor):
                the origins of rays, of shape :math:`(\text{num_rays}, 3)` or
                :math:`(\text{num_cameras}, \text{num_rays}, 3)`
            ray_dir (torch.Tensor):
                the directions of rays, of shape :math:`(\text{num_rays}, 3)` or
                :math:`(\text{num_cameras}, \text{num_rays}, 3)`

        Returns:
            (torch.Tensor, torch.Tensor):
                the transformed ray origins and directions, of same shape than inputs
        """
        num_cameras = 1           # C - number of cameras
        batch_size = ray_dir.shape[-2]    # B - number of vectors
        d = ray_dir.expand(num_cameras, batch_size, 3)[..., None]   # Expand as (C, B, 3, 1)
        o = ray_orig.expand(num_cameras, batch_size, 3)[..., None]  # Expand as (C, B, 3, 1)
        R = self.R[:, None].expand(num_cameras, batch_size, 3, 3)   # Expand as (C, B, 3, 3)
        t = self.t[:, None].expand(num_cameras, batch_size, 3, 1)   # Expand as (C, B, 3, 1)
        R_T = R.transpose(2, 3)     # Transforms orientation from camera to world
        transformed_dir = R_T @ d   # Inverse rotation is transposition: R^(-1) = R^T
        transformed_orig = R_T @ (o - t)
        return transformed_orig.squeeze(-1), transformed_dir.squeeze(-1)



def finitediff_gradient(x, f, eps=0.005):
    """Compute 3D gradient using finite difference.

    Args:
        x (torch.FloatTensor): Coordinate tensor of shape [..., 3]
        f (nn.Module): The function to perform autodiff on.
    """
    eps_x = torch.tensor([eps, 0.0, 0.0], device=x.device)
    eps_y = torch.tensor([0.0, eps, 0.0], device=x.device)
    eps_z = torch.tensor([0.0, 0.0, eps], device=x.device)

    grad = torch.cat([f(x + eps_x) - f(x - eps_x),
                      f(x + eps_y) - f(x - eps_y),
                      f(x + eps_z) - f(x - eps_z)], dim=-1)
    grad = grad / (eps * 2.0)

    return grad




def sphere_trace_sdf(nef, ray_o, ray_d, img, calib, num_steps = 1024, step_size = 0.8, min_dis=0.0003, camera_clamp = 10.0 ):
    """PyTorch implementation of sphere tracing."""

    # Distanace from ray origin
    t = torch.zeros(ray_o.shape[0], 1, device=ray_o.device)

    # Position in model space
    # save the ray_o and ray_d as an image using opencv
    import opencv as cv2


    
    x = torch.addcmul(ray_o, ray_d, t)
    x = torch.transpose(x, 0, 1)
    x = x.unsqueeze(0) 

    cond = torch.ones_like(t).bool()[:,0]
    
    normal = torch.zeros_like(x)
    # This function is in fact differentiable, but we treat it as if it's not, because
    # it evaluates a very long chain of recursive neural networks (essentially a NN with depth of
    # ~1600 layers or so). This is not sustainable in terms of memory use, so we return the final hit
    # locations, where additional quantities (normal, depth, segmentation) can be determined. The
    # gradients will propagate only to these locations. 
    with torch.no_grad():
        img = img.to(x.device)
        calib = calib.to(x.device)

        d= nef(img, x, calib)
        
        dprev = d.clone()

        # If cond is TRUE, then the corresponding ray has not hit yet.
        # OR, the corresponding ray has exit the clipping plane.
        #cond = torch.ones_like(d).bool()[:,0]

        # If miss is TRUE, then the corresponding ray has missed entirely.
        hit = torch.zeros_like(d).byte()
        
        for i in range(num_steps):
            # 1. Check if ray hits.
            #hit = (torch.abs(d) < self._MIN_DIS)[:,0] 
            # 2. Check that the sphere tracing is not oscillating
            #hit = hit | (torch.abs((d + dprev) / 2.0) < self._MIN_DIS * 3)[:,0]
            
            # 3. Check that the ray has not exit the far clipping plane.
            #cond = (torch.abs(t) < self.clamp[1])[:,0]
            hit = (torch.abs(t) < camera_clamp)[:,0]
            
            # 1. not hit surface
            cond = cond & (torch.abs(d) > min_dis)[:,0] 

            # 2. not oscillating
            cond = cond & (torch.abs((d + dprev) / 2.0) > min_dis * 3)[:,0]
            
            # 3. not a hit
            cond = cond & hit
            
            #cond = cond & ~hit
            
            # If the sum is 0, that means that all rays have hit, or missed.
            if not cond.any():
                break

            # Advance the x, by updating with a new t
            x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)
            
            # Store the previous distance
            dprev = torch.where(cond.unsqueeze(1), d, dprev)

            # Update the distance to surface at x
            d[cond], _ = nef(img, x[cond], calib) * step_size

            # Update the distance from origin 
            t = torch.where(cond.view(cond.shape[0], 1), t+d, t)

    # AABB cull 

    hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)
    #hit = torch.ones_like(d).byte()[...,0]
    
    # The function will return 
    #  x: the final model-space coordinate of the render
    #  t: the final distance from origin
    #  d: the final distance value from
    #  miss: a vector containing bools of whether each ray was a hit or miss
    

    """if hit.any():
        grad = finitediff_gradient(x[hit], nef)
        _normal = F.normalize(grad, p=2, dim=-1, eps=1e-5)
        normal[hit] = _normal"""
    
    print(x)
    print(hit)
    return x, t, d, hit
        

def sphere_tracing(nef, ray_o, ray_d, calib, device, num_steps = 64, step_size = 0.05, min_dis=0.0003, camera_clamp = 3.0, calculate_normal = False):
    # Distanace from ray origin
    t = torch.zeros(ray_o.shape[0], 1, device=ray_o.device)
    # Position in model space
    x = torch.addcmul(ray_o, ray_d, t)

    x = torch.transpose(x, 0, 1)
    x = x.unsqueeze(0) 

    cond = torch.ones_like(t).bool()[:,0]
    normal = torch.zeros_like(x)
    # This function is in fact differentiable, but we treat it as if it's not, because
    # it evaluates a very long chain of recursive neural networks (essentially a NN with depth of
    # ~1600 layers or so). This is not sustainable in terms of memory use, so we return the final hit
    # locations, where additional quantities (normal, depth, segmentation) can be determined. The
    # gradients will propagate only to these locations. 
    with torch.no_grad():
       
        calib = calib.to(device)

        nef.query(x, calib)
        #d = nef.get_preds()
        d = nef.get_preds()[0][0]
        d = (d -0.5)*-2
        
        
        dprev = d.clone()
        # If cond is TRUE, then the corresponding ray has not hit yet.
        # OR, the corresponding ray has exit the clipping plane.
        #cond = torch.ones_like(d).bool()[:,0]
        # If miss is TRUE, then the corresponding ray has missed entirely.
        hit = torch.zeros_like(d).byte()    

       
        d_list = []
        t_list = []
        x_list = []
        y_list = []
        z_list = []


        for i in range(num_steps):
            # calculate max and min of d,t and x, convert them to numpy and append to list
            d_list.append([torch.max(d).cpu().numpy(), torch.min(d).cpu().numpy()])
            t_list.append([torch.max(t[:,0]).cpu().numpy(), torch.min(t[:,0]).cpu().numpy()])
            x_list.append([torch.max(x[0,0,:]).cpu().numpy(), torch.min(x[0,0,:]).cpu().numpy()])
            y_list.append([torch.max(x[0,1,:]).cpu().numpy(), torch.min(x[0,1,:]).cpu().numpy()])
            z_list.append([torch.max(x[0,2,:]).cpu().numpy(), torch.min(x[0,2,:]).cpu().numpy()])
            
            # 1. Check if ray hits.
            #hit = (torch.abs(d) < self._MIN_DIS)[:,0] 
            # 2. Check that the sphere tracing is not oscillating
            #hit = hit | (torch.abs((d + dprev) / 2.0) < self._MIN_DIS * 3)[:,0]
            
            # 3. Check that the ray has not exit the far clipping plane.
            #cond = (torch.abs(t) < self.clamp[1])[:,0]
            
            
            hit = (torch.abs(t) < camera_clamp)[:,0]
            """print("###################")
            if hit.all():
                print("hit all true")
            elif hit.any():
                print("hit true and false")
            else:
                print("hit all false")
            print("d first elements: ", d[0:10])
            print("cond shape: ", cond.shape)
            print("hit shape: ", hit.shape)
            print("d shape: ", d.shape)
            print("dprev shape: ", dprev.shape)
            print("t shape: ", t.shape)
            print("x shape: ", x.shape)
            print("ray_o shape: ", ray_o.shape)
            print("ray_d shape: ", ray_d.shape)
            print("###################")"""
            
            # 1. not hit surface
            cond = cond & (torch.abs(d) > min_dis)
            # 2. not oscillating
            cond = cond & (torch.abs((d + dprev) / 2.0) > min_dis * 3)
            
            # 3. not a hit
            cond = cond & hit
            
            """print(cond)
            if cond.all():
                print("all true")
            elif cond.any():
                print("true and false")
            else:
                print("all false")"""
            
            #cond = cond & ~hit
            
            # If the sum is 0, that means that all rays have hit, or missed.
            if not cond.any():
                break
            # Advance the x, by updating with a new t
            x_new = torch.addcmul(ray_o, ray_d, t)
            x_new = torch.transpose(x_new, 0, 1)
            x_new = x_new.unsqueeze(0) 
            """print("x_new shape: ", x_new.shape)"""
            # change the values pf x where cond is true with the values of x_new
            # create a new tensor with the indexes of the cond tensor for which the value is true
            indexes = torch.nonzero(cond)
            x[0,:,indexes] = x_new[0,:,indexes]
            dprev = torch.where(cond, d, dprev)
            # Update the distance to surface at x
            nef.query(x, calib)
            #d = nef.get_preds()
            d[cond] = nef.get_preds()[0][0][cond]
            d[cond] = (d[cond] -0.5)*-2* step_size

            # Update the distance from origin 
            t[indexes,0] = t[indexes,0] + d[indexes]
            

    # AABB cull 
    x = torch.transpose(x, 1, 2)
    
    """print(x)
    print(x.shape)
    print(x.min(dim=1))
    print(x.max(dim=1))
    if hit.all():
        print("hit all true")
    elif hit.any():
        print("hit true and false")
    else:
                print("hit all false")"""

    hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)
    #hit = torch.ones_like(d).byte()[...,0]
    
    # The function will return 
    #  x: the final model-space coordinate of the render
    #  t: the final distance from origin
    #  d: the final distance value from
    #  miss: a vector containing bools of whether each ray was a hit or miss
    normal = None
    if calculate_normal:
        if hit.any():
            grad = finitediff_gradient(x[hit], nef.get_forward_function("sdf"))
            _normal = F.normalize(grad, p=2, dim=-1, eps=1e-5)
            normal[hit] = _normal
    
    # x,y,z,d and t lists create a plot with two lines, one per min and one per max values, then save the plot in the folder
    """import matplotlib.pyplot as plt
    # d_list is a list of tuples containing two numbers each, use the first number of each tuple to create a list and the same for the second
    d_list1 = [item[0] for item in d_list]
    print("AO")
    d_list2 = [item[1] for item in d_list]
    # use d_list1 and d_list2 to create a plot with two lines and save it
    plt.plot(d_list1, label="max")
    plt.plot(d_list2, label="min")
    plt.legend()
    plt.title("d")
    plt.xlabel("Value")
    plt.ylabel("Iterations")
    plt.savefig('d_list.png')
    plt.close()
    t_list1 = [item[0] for item in t_list]
    t_list2 = [item[1] for item in t_list]
    plt.plot(t_list1, label="max")
    plt.plot(t_list2, label="min")
    plt.legend()
    plt.title("t")
    plt.xlabel("Value")
    plt.ylabel("Iterations")
    plt.savefig('t_list.png')
    plt.close()
    x_list1 = [item[0] for item in x_list]
    x_list2 = [item[1] for item in x_list]
    plt.plot(x_list1, label="max")
    plt.plot(x_list2, label="min")
    plt.legend()
    plt.title("x")
    plt.xlabel("Value")
    plt.ylabel("Iterations")
    plt.savefig('x_list.png')
    plt.close()
    y_list1 = [item[0] for item in y_list]
    y_list2 = [item[1] for item in y_list]
    plt.plot(y_list1)
    plt.plot(y_list2)
    plt.legend()
    plt.title("y")
    plt.xlabel("Value")
    plt.ylabel("Iterations")
    plt.savefig('y_list.png')
    plt.close()
    z_list1 = [item[0] for item in z_list]
    z_list2 = [item[1] for item in z_list]
    plt.plot(z_list1, label="max")
    plt.plot(z_list2, label="min")
    plt.legend()
    plt.title("z")
    plt.xlabel("Value")
    plt.ylabel("Iterations")
    plt.savefig('z_list.png')
    plt.close()"""

    """if hit.all():
        print("hit all true")
    elif hit.any():
        print("hit true and false")
    else:
                print("hit all false")
    # count the number of False in in hit and print it
    print("GIGI d:")
    print(d)
    # count the number of True that there are in hit and print it
    print("GIGI hit:")
    print(hit.sum())"""

    # create a copy of hit called cpy, where cpy is True change it to 255, 0 otherwise. then reshape it from [512x512] to [512,512,1]
    """cpy = hit.clone()
    cpy = cpy.reshape(512,512)
    img_mask = torch.zeros([512,512,3], dtype=torch.uint8)
    img_mask[cpy] = 255
    # save cpy as an image using PIL
    
    img_mask = img_mask.cpu().numpy()
    from PIL import Image
    img_mask = Image.fromarray(img_mask)
    img_mask.save("hit.png")"""


    return x, t, d, hit, normal

