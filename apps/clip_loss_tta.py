from datetime import datetime
import math
import sys
import os

from rembg.bg import remove
import numpy as np
import io
import cv2
from PIL import Image
import sys
import clip

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


from rembg.session_factory import new_session

import cv2
import time
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.models import detection

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

import trimesh

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# get options
opt = BaseOptions().parse()

from camera.camera import Camera
from raygen import generate_centered_pixel_coords, generate_pinhole_rays, generate_ortho_rays
from sphere_tracing import sphere_tracing


class MeshEvaluator:
    _normal_render = None

    def __init__(self):
        pass

    @staticmethod
    def init_gl():
        from lib.renderer.gl.normal_render import NormalRender
        from lib.renderer.gl.init_gl import initialize_GL_context
        initialize_GL_context(width=512, height=512, egl=True)
        MeshEvaluator._normal_render = NormalRender(width=512, height=512)


    def set_mesh(self, src_path, tgt_path, scale_factor=1.0, offset=0):
        self.src_mesh = trimesh.load(src_path)
        self.tgt_mesh = trimesh.load(tgt_path)

        self.scale_factor = scale_factor
        self.offset = offset


    def euler_to_rot_mat(self, r_x, r_y, r_z):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(r_x), -math.sin(r_x)],
                        [0, math.sin(r_x), math.cos(r_x)]
                        ])

        R_y = np.array([[math.cos(r_y), 0, math.sin(r_y)],
                        [0, 1, 0],
                        [-math.sin(r_y), 0, math.cos(r_y)]
                        ])

        R_z = np.array([[math.cos(r_z), -math.sin(r_z), 0],
                        [math.sin(r_z), math.cos(r_z), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R

    def _render_normal(self, mesh, deg):
        view_mat = np.identity(4)
        view_mat[:3, :3] *= 2 / 256
        rz = deg / 180. * np.pi
        model_mat = np.identity(4)
        model_mat[:3, :3] = self.euler_to_rot_mat(0, rz, 0)
        model_mat[1, 3] = self.offset
        view_mat[2, 2] *= -1

        self._normal_render.set_matrices(view_mat, model_mat)
        self._normal_render.set_normal_mesh(self.scale_factor*mesh.vertices, mesh.faces, mesh.vertex_normals, mesh.faces)
        self._normal_render.draw()
        normal_img = self._normal_render.get_color()
        return normal_img

    def _get_reproj_normal_error(self, deg):
        tgt_normal = self._render_normal(self.tgt_mesh, deg)
        src_normal = self._render_normal(self.src_mesh, deg)

        error = ((src_normal[:, :, :3] - tgt_normal[:, :, :3]) ** 2).mean() * 3

        return error, src_normal, tgt_normal

    def get_reproj_normal_error(self, frontal=True, back=True, left=True, right=True, save_demo_img=None):
        print(f"save_demo: {save_demo_img}")
        # reproj error
        # if save_demo_img is not None, save a visualization at the given path (etc, "./test.png")
        if self._normal_render is None:
            print("In order to use normal render, "
                  "you have to call init_gl() before initialing any evaluator objects.")
            return -1

        side_cnt = 0
        total_error = 0
        demo_list = []
        if frontal:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(0)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if back:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(180)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if left:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(90)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if right:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(270)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if save_demo_img is not None:
            res_array = np.concatenate(demo_list, axis=1)
            #res_img = Image.fromarray((res_array * 255).astype(np.uint8))
            res_img = Image.fromarray((tgt_normal * 255).astype(np.uint8))
            res_img.save(save_demo_img)
        return total_error / side_cnt


    def get_chamfer_dist(self, num_samples=10000):
        # Chamfer
        src_surf_pts, _ = trimesh.sample.sample_surface(self.src_mesh, num_samples)
        tgt_surf_pts, _ = trimesh.sample.sample_surface(self.tgt_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(self.tgt_mesh, src_surf_pts)
        _, tgt_src_dist, _ = trimesh.proximity.closest_point(self.src_mesh, tgt_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0
        tgt_src_dist[np.isnan(tgt_src_dist)] = 0

        src_tgt_dist = src_tgt_dist.mean()
        tgt_src_dist = tgt_src_dist.mean()

        chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

        return chamfer_dist

    def get_surface_dist(self, num_samples=10000):
        # P2S
        src_surf_pts, _ = trimesh.sample.sample_surface(self.src_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(self.tgt_mesh, src_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0

        src_tgt_dist = src_tgt_dist.mean()

        return src_tgt_dist


class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))
            print('net G loaded ...', opt.load_netC_checkpoint_path)

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

        # def constants(self):# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/constants.py
        self.FOCAL_LENGTH = 5000.
        self.IMG_RES = 224

        # Mean and standard deviation for normalizing input image
        self.IMG_NORM_MEAN = [0.485, 0.456, 0.406]
        self.IMG_NORM_STD = [0.229, 0.224, 0.225]
        """
        We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
        We keep a superset of 24 joints such that we include all joints from every dataset.
        If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
        The joints used here are the following:
        """
        self.JOINT_NAMES = [
            # 25 OpenPose joints (in the order provided by OpenPose)
            'OP Nose',
            'OP Neck',
            'OP RShoulder',
            'OP RElbow',
            'OP RWrist',
            'OP LShoulder',
            'OP LElbow',
            'OP LWrist',
            'OP MidHip',
            'OP RHip',
            'OP RKnee',
            'OP RAnkle',
            'OP LHip',
            'OP LKnee',
            'OP LAnkle',
            'OP REye',
            'OP LEye',
            'OP REar',
            'OP LEar',
            'OP LBigToe',
            'OP LSmallToe',
            'OP LHeel',
            'OP RBigToe',
            'OP RSmallToe',
            'OP RHeel',
            # 24 Ground Truth joints (superset of joints from different datasets)
            'Right Ankle',
            'Right Knee',
            'Right Hip',    # 2
            'Left Hip',
            'Left Knee',    # 4
            'Left Ankle',
            'Right Wrist',    # 6
            'Right Elbow',
            'Right Shoulder',    # 8
            'Left Shoulder',
            'Left Elbow',    # 10
            'Left Wrist',
            'Neck (LSP)',    # 12
            'Top of Head (LSP)',
            'Pelvis (MPII)',    # 14
            'Thorax (MPII)',
            'Spine (H36M)',    # 16
            'Jaw (H36M)',
            'Head (H36M)',    # 18
            'Nose',
            'Left Eye',
            'Right Eye',
            'Left Ear',
            'Right Ear'
        ]

        # Dict containing the joints in numerical order
        self.JOINT_IDS = {self.JOINT_NAMES[i]: i for i in range(len(self.JOINT_NAMES))}

        # Map joints to SMPL joints
        self.JOINT_MAP = {
            'OP Nose': 24,
            'OP Neck': 12,
            'OP RShoulder': 17,
            'OP RElbow': 19,
            'OP RWrist': 21,
            'OP LShoulder': 16,
            'OP LElbow': 18,
            'OP LWrist': 20,
            'OP MidHip': 0,
            'OP RHip': 2,
            'OP RKnee': 5,
            'OP RAnkle': 8,
            'OP LHip': 1,
            'OP LKnee': 4,
            'OP LAnkle': 7,
            'OP REye': 25,
            'OP LEye': 26,
            'OP REar': 27,
            'OP LEar': 28,
            'OP LBigToe': 29,
            'OP LSmallToe': 30,
            'OP LHeel': 31,
            'OP RBigToe': 32,
            'OP RSmallToe': 33,
            'OP RHeel': 34,
            'Right Ankle': 8,
            'Right Knee': 5,
            'Right Hip': 45,
            'Left Hip': 46,
            'Left Knee': 4,
            'Left Ankle': 7,
            'Right Wrist': 21,
            'Right Elbow': 19,
            'Right Shoulder': 17,
            'Left Shoulder': 16,
            'Left Elbow': 18,
            'Left Wrist': 20,
            'Neck (LSP)': 47,
            'Top of Head (LSP)': 48,
            'Pelvis (MPII)': 49,
            'Thorax (MPII)': 50,
            'Spine (H36M)': 51,
            'Jaw (H36M)': 52,
            'Head (H36M)': 53,
            'Nose': 24,
            'Left Eye': 26,
            'Right Eye': 25,
            'Left Ear': 28,
            'Right Ear': 27
        }

        # Joint selectors
        # Indices to get the 14 LSP joints from the 17 H36M joints
        self.H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
        self.H36M_TO_J14 = self.H36M_TO_J17[:14]
        # Indices to get the 14 LSP joints from the ground truth joints
        self.J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
        self.J24_TO_J14 = self.J24_TO_J17[:14]
        self.J24_TO_J19 = self.J24_TO_J17[:14] + [19, 20, 21, 22, 23]
        self.J24_TO_JCOCO = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

        # Permutation of SMPL pose parameters when flipping the shape
        self.SMPL_JOINTS_FLIP_PERM = [
            0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22
        ]
        self.SMPL_POSE_FLIP_PERM = []
        for i in self.SMPL_JOINTS_FLIP_PERM:
            self.SMPL_POSE_FLIP_PERM.append(3 * i)
            self.SMPL_POSE_FLIP_PERM.append(3 * i + 1)
            self.SMPL_POSE_FLIP_PERM.append(3 * i + 2)
        # Permutation indices for the 24 ground truth joints
        self.J24_FLIP_PERM = [
            5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22
        ]
        # Permutation indices for the full set of 49 joints
        self.J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
            + [25+i for i in self.J24_FLIP_PERM]
        self.SMPL_J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
            + [25+i for i in self.SMPL_JOINTS_FLIP_PERM]

    def get_transformer(self, input_res):

        image_to_tensor = transforms.Compose(
            [
                transforms.Resize(input_res),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        mask_to_tensor = transforms.Compose(
            [
                transforms.Resize(input_res),
                transforms.ToTensor(),
                transforms.Normalize((0.0, ), (1.0, ))
            ]
        )

        image_to_pymaf_tensor = transforms.Compose(
            [
                transforms.Resize(size=224),
                transforms.Normalize(mean=self.IMG_NORM_MEAN, std=self.IMG_NORM_STD)
            ]
        )

        image_to_pixie_tensor = transforms.Compose([transforms.Resize(224)])

        def image_to_hybrik_tensor(img):
            # mean
            img[0].add_(-0.406)
            img[1].add_(-0.457)
            img[2].add_(-0.480)

            # std
            img[0].div_(0.225)
            img[1].div_(0.224)
            img[2].div_(0.229)
            return img

        return [
            image_to_tensor, mask_to_tensor, image_to_pymaf_tensor, image_to_pixie_tensor,
            image_to_hybrik_tensor
        ]

    def load_img(self, img_file):

        if img_file.endswith("exr"):
            img = cv2.imread(img_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
        else :
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        # considering non 8-bit image
        if img.dtype != np.uint8 :
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if not img_file.endswith("png"):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        return img


    def get_affine_matrix(self, center, translate, scale):
        cx, cy = center
        tx, ty = translate

        M = [1, 0, 0, 0, 1, 0]
        M = [x * scale for x in M]

        # Apply translation and of center translation: RSS * C^-1
        M[2] += M[0] * (-cx) + M[1] * (-cy)
        M[5] += M[3] * (-cx) + M[4] * (-cy)

        # Apply center translation: T * C * RSS * C^-1
        M[2] += cx + tx
        M[5] += cy + ty
        return M

    def aug_matrix(self, w1, h1, w2, h2):
        dx = (w2 - w1) / 2.0
        dy = (h2 - h1) / 2.0

        matrix_trans = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])

        scale = np.min([float(w2) / w1, float(h2) / h1])

        M = self.get_affine_matrix(center=(w2 / 2.0, h2 / 2.0), translate=(0, 0), scale=scale)

        M = np.array(M + [0., 0., 1.]).reshape(3, 3)
        M = M.dot(matrix_trans)

        return M

    def crop_for_hybrik(self, img, center, scale):
        inp_h, inp_w = (256, 256)
        trans = self.get_affine_transform(center, scale, 0, [inp_w, inp_h])
        new_img = cv2.warpAffine(img, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        return new_img


    def get_affine_transform(
        self, center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
    ):
        def get_dir(src_point, rot_rad):
            """Rotate the point by `rot_rad` degree."""
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)

            src_result = [0, 0]
            src_result[0] = src_point[0] * cs - src_point[1] * sn
            src_result[1] = src_point[0] * sn + src_point[1] * cs

            return src_result

        def get_3rd_point(a, b):
            """Return vector c that perpendicular to (a - b)."""
            direct = a - b
            return b + np.array([-direct[1], direct[0]], dtype=np.float32)

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
    
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    
    def load_image(self, image_path): # , mask_path):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        # Mask
        # mask = Image.open(mask_path).convert('L')
        # mask = transforms.Resize(self.load_size)(mask)
        # mask = transforms.ToTensor()(mask).float()
        # image
        image = Image.open(image_path).convert('RGB')
        
        mask = remove(image, alpha_matting=True, session=new_session("u2netP"))

            
        # from mask only get the alpha channel
        mask = mask.split()[-1]
        
        # mask.save("/home/fantini/PIFu/out.png")
        image = self.to_tensor(image)
        
        
        mask = transforms.Resize(self.load_size)(mask)
        
        mask = transforms.ToTensor()(mask).float()

        #mask = self.create_mask(image_path)
        image = mask.expand_as(image) * image
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }
    
    def render(self, data, output_path):
        def set_train():
            self.netG.eval()
            self.netC.train()

        def set_eval():
            self.netG.eval()
            self.netC.eval()
        clip_encoder, _ = clip.load(self.opt.clip_model_name, device = self.cuda)
        self.clip_encoder = clip_encoder.eval().requires_grad_(False).cuda()
        self.clip_normalizer = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.clip_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                self.clip_normalizer
            ])

        def get_clip_feature( x):
            transf_x = self.clip_transform(x)
            feat = self.clip_encoder.encode_image(transf_x)
            return feat

        
        lr = 1e-4
        optimizerC = torch.optim.Adam(self.netC.parameters(), lr=lr)
        num_iters = 100
        set_eval()
        data['calib'] = data['calib'].to(self.cuda)
        data['img'] = data['img'].to(self.cuda)
        with torch.no_grad():
            self.netG.filter(data['img'])
            self.netC.filter(data['img'])
            self.netC.attach(self.netG.get_im_feat())

        look_at = torch.zeros( (4, 3), dtype=torch.float32, device=self.cuda)
        camera_position = torch.tensor( [ [0, 0, 2],
                                        [2, 0, 0],
                                        [0, 0, -2],
                                        [-2, 0, 0]  ]  , dtype=torch.float32, device=self.cuda)
        camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=self.cuda).repeat(4, 1,)

        data['img']=  data['img']*0.5 + 0.5

        points_on_surfaces = []
        hits = []
        print("\nRendering 4 views:")
        for i in tqdm.tqdm(range(4)):   
            with torch.no_grad():    
                camera = Camera.from_args(eye=camera_position[i],
                                            at=look_at[0],
                                            up=camera_up_direction[0],
                                            fov=10,
                                            width=512,
                                            height=512,
                                            dtype=torch.float32)

                ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                                camera.width, camera.height, device=self.cuda)

                
                ray_orig, ray_dir = generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid)
                #ray_orig, ray_dir = generate_ortho_rays(camera.to(ray_grid[0].device), ray_grid)

                
                
                
                

                points_on_surface, _ ,_ , hit, _ = sphere_tracing(self.netG,ray_orig,ray_dir,data['calib'],
                                                            device=self.cuda )
                
                points_on_surface = points_on_surface[hit].unsqueeze(0)

                points_on_surface = torch.transpose(points_on_surface, 1, 2)
                points_on_surface = points_on_surface.to(self.cuda)


                points_on_surfaces.append(points_on_surface)
                hits.append(hit)
                
        print("\nFinetuning:")
        for iteration in tqdm.tqdm(range(num_iters)):
            set_train()
            optimizerC.zero_grad()
            i = iteration % 4    

            color = torch.zeros([262144,3], device = self.cuda)
            rgb , _ = self.netC.forward(data['img'], None, points_on_surfaces[i],  data['calib'], None, only_query = True )
            rgb = rgb[0]*0.5 + 0.5
            
            
            color[hits[i][0,:]] = rgb.T
            color = color.reshape(512,512,3)
            color = torch.permute(color, (2,0,1))
            color = color.unsqueeze(0)

            target_clip = get_clip_feature(data['img'])
            rendered_clip = get_clip_feature(color)
            cosine = torch.cosine_similarity(torch.mean(target_clip, dim=0), torch.mean(rendered_clip, dim=0), dim=0)

            clip_loss = 1.0 - cosine
            
            
            """rendering = Image.fromarray(color.astype(np.uint8))
            rendering.save(output_path + f"/out_diff_rendering_{i}.png")
            print("RENDERING DONE")"""
           
            clip_loss.backward()
            optimizerC.step()
            directions = ["front", "right","back","left"]
           
            print(f"### Iteration {iteration} - {directions[i]} camera - cosine distance: {clip_loss.item()}")

            rendered_image = np.zeros([262144,3])
            rgb_copy = torch.tensor(rgb, requires_grad = False)
            rgb_copy = rgb_copy.cpu().numpy()
            rendered_image[hits[i].cpu().numpy()[0,:]] = rgb_copy.T
            rendered_image = rendered_image*255
            rendered_image = rendered_image.reshape(512,512,3)
            rendering = Image.fromarray(rendered_image.astype(np.uint8))
            index = "{:04d}".format(iteration)
            rendering.save(output_path + f"/{index}_{directions[i]}_render.png")
    

    def eval(self, data, use_octree=False, path=None):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        

        self.render(data, output_path="/scratch/izar/fantini/renderings_clip")
        



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore") 
    evaluator = Evaluator(opt)
    mesh_evaluator = MeshEvaluator()
    mesh_evaluator.init_gl()
    
    print("test folder path: ", opt.test_folder_path)
    all_files = glob.glob(os.path.join(opt.test_folder_path, '*'))
    eval_objs = ['0029', '0031', '0049', '0101', '0109'] # hard coded for now; testing on 5 objects
    #test_images = [f for f in all_files if ('png' in f or 'jpg' in f) and (not 'mask' in f) and any(obj in f for obj in eval_objs)]
    test_images = [f for f in all_files if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    rendered_images = []
    mask_images = []

    test_images.sort()

    print("num; ", len(test_images))


    total_vals = []
    items = []
    
    # get a timestamp for the results folder and make it a string
    now = datetime.now()

    folder_name = now.strftime("day_%Y_%m_%d_time_%H_%M_%S")
    print("date and time:",folder_name)

    results_path = f"/scratch/izar/fantini/final/results"
    #results_path = f"/scratch/izar/fantini/final/results/{folder_name}"
    # if folder doesn't exist, create it
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        

    for image_path in tqdm.tqdm(test_images): # , test_masks)):
            
        print(image_path) # , mask_path)
        data = evaluator.load_image(image_path) # , mask_path)
        evaluator.eval(data, True, results_path)
        # metrics calculation
        reconstructed_obj_path = '%s/%s/result_%s.obj' % (results_path, opt.name, data['name'])
        print(f"reconstructed_obj_path: {reconstructed_obj_path}")
        # from all_files get the obj file with the same code as the image, getting it by splitting the path name by "_" and getting the -3 element
        test_obj = [f for f in all_files if 'obj' in f and image_path.split("_")[-3] in f][0]
        print(f"test_obj: {test_obj}")
        mesh_evaluator.set_mesh(reconstructed_obj_path, test_obj)
        vals = []
        vals.append(0.1 * mesh_evaluator.get_chamfer_dist(500))
        vals.append(0.1 * mesh_evaluator.get_surface_dist(500))
        vals.append(4.0 * mesh_evaluator.get_reproj_normal_error(save_demo_img=os.path.join(results_path, data['name'] + '_try.png')))
        print(f"vals: {vals}")
        item = {
                'name': '%s' % (image_path),
                'vals': vals
            }
        total_vals.append(vals)
        items.append(item)
        np.save(os.path.join(results_path, 'rp-item.npy'), np.array(items))
        np.save(os.path.join(results_path, 'rp-vals.npy'), total_vals)
