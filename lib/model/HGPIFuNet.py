import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
import clip
import pdb
from torchvision import transforms
from .Multimodal import CLIP_Transform, WrapperModel, DataParallelModel, PerceiverResampler, PatchEmbedding
from .dpt import DPTRegressionModel
from einops import rearrange


class HGPIFuNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'hgpifu'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)

        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())

        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        # We also use this to store CLIP features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []
        if opt.use_clip_encoder:
            self.clip_encoder, _ = clip.load(opt.clip_model_name)
            self.clip_encoder = self.clip_encoder.eval().requires_grad_(False)
            self.clip_normalizer = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            self.clip_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                self.clip_normalizer
            ])
            self.clip_dim_dict = {
                'ViT-L/14': 768,
                'RN50': 512,
            }
            if self.opt.feature_fusion == 'tf_concat':
                # transform then concat
                self.clip_feature_transform = CLIP_Transform(self.clip_dim_dict[opt.clip_model_name])
            elif self.opt.feature_fusion == 'concat':
                pass
            elif self.opt.feature_fusion == 'add':
                pass
            elif self.opt.feature_fusion == 'prismer':
                self.expert_fusion = PerceiverResampler(width=128, layers=2, heads=8, num_latents=128)
                self.pifu_patchify = PatchEmbedding(patch_size=16, in_channels=3, embedding_dim=128) 
                self.dpt_patchify = PatchEmbedding(patch_size=16, in_channels=3, embedding_dim=128)
                self.clip_proj = nn.Linear(self.clip_dim_dict[opt.clip_model_name], 128)
                
        
        if opt.use_dpt:
            map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
            model_arch_dpt = DPTRegressionModel(num_channels = 3, backbone = 'vitb_rn50_384', non_negative=False)
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # self.dpt = WrapperModel(DataParallelModel(model_arch_dpt.to(device)))
            self.dpt = WrapperModel(DataParallelModel(model_arch_dpt))
            model_robust_state_dict = torch.load(self.opt.dpt_path, map_location=map_location)
            self.dpt.load_state_dict(model_robust_state_dict["('rgb', '"+'normal'+"')"])
            self.dpt = self.dpt.eval().requires_grad_(False)
            self.dpt_tf = transforms.Compose([transforms.Resize(256, antialias=True), transforms.CenterCrop(256)])
            
        init_net(self, gpu_ids=self.opt.gpu_ids)

    
    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        if self.opt.use_clip_encoder:
            # First transfrom images to clip input
            clip_tf = self.clip_transform(images)
            clip_feature = self.clip_encoder.encode_image(clip_tf) #[bz, 128]
            if self.opt.feature_fusion == 'tf_concat':
                # transform from [bz, 256, 3] to [bz, 256, 128, 128]
                transformed_clip = self.clip_feature_transform(clip_feature.float())
                self.im_feat_list.append(transformed_clip)
                
            elif self.opt.feature_fusion == 'add':
                # add clip feature to each intermediate feature
                # shape transform from [bz, 256, 3] to [bz, 256, 128, 128]
                clip_feature_tf = clip_feature.reshape(clip_feature.shape[0], 256, -1)
                clip_feature_tf = torch.mean(clip_feature_tf, dim=-1)
                for i in range(len(self.im_feat_list)): # [bz, 256, 1, 1]
                    self.im_feat_list[i] = self.im_feat_list[i] + clip_feature_tf.unsqueeze(-1).unsqueeze(-1)
            elif self.opt.feature_fusion == 'prismer':
                patchify_pifu = self.pifu_patchify(images) # [bz, 3, 512, 512] --> [16, 1024, 128]
                if self.opt.use_dpt:
                    normal = self.dpt(self.dpt_tf(images)) 
                    pathchify_dpt = self.dpt_patchify(normal) # [16, 256, 128]
                    clip_feature = self.clip_proj(clip_feature.float()).unsqueeze(1) # [bz, 128] --> [bz, 1, 128]
                    experts_input = rearrange(torch.cat([patchify_pifu, pathchify_dpt, clip_feature], dim=1), 'b l d -> l b d') # [1281, 16, 128]
                elif self.opt.prismer_only_clip:
                    clip_feature = self.clip_proj(clip_feature.float()).unsqueeze(1) # [bz, 128] --> [bz, 1, 128]
                    experts_input = rearrange(torch.cat([clip_feature], dim=1), 'b l d -> l b d') 
                else:
                    clip_feature = self.clip_proj(clip_feature.float()).unsqueeze(1) # [bz, 128] --> [bz, 1, 128]
                    experts_input = rearrange(torch.cat([patchify_pifu, clip_feature], dim=1), 'b l d -> l b d') # [1025, 16, 128]
                experts_output = self.expert_fusion(experts_input) # [128, bz, 128]
                experts_output = rearrange(experts_output, 'l b d -> b l d').unsqueeze(1) # [bz, 1, 128, 128]
                self.im_feat_list = [torch.cat([self.im_feat_list[-1], experts_output], dim=1)] # [bz, 257, 128, 128]
                

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        z_feat = self.normalizer(z, calibs=calibs)

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []
        for im_feat in self.im_feat_list: # im_feat [2, 256, 128, 128]
            # [B, Feat_i + z, N]
            point_local_feat_list = [self.index(im_feat, xy), z_feat]

            if self.opt.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0
            pred = in_img[:,None].float() * self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        # pdb.set_trace()
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term(preds, self.labels)
        error /= len(self.intermediate_preds_list)
        
        return error
    
    def forward(self, images, points, calibs, transforms=None, labels=None):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels)

        # get the prediction
        res = self.get_preds()
        
        # get the error
        error = self.get_error()

        return res, error