import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv2d_bn_relu, down_block2d, res_mod_block2d, up_block2d
from .layers import conv2d
from .utils import make_coordinate_grid


class Generator(nn.Module):
    def __init__(self, num_affines, num_residual_mod_blocks, using_gaussian):
        super(Generator, self).__init__()
        self.using_gaussian = using_gaussian

        num_input_channel = (num_affines + 1) * 3
        if self.using_gaussian:
            num_input_channel += num_affines

        self.encoder = nn.Sequential(
            conv2d_bn_relu(num_input_channel, 64, 7, 1, 3),
            down_block2d(64, 128),
            down_block2d(128, 256),
            down_block2d(256, 512)
        )
        self.mask_predictor = conv2d(512, num_affines+1, 5, 1, 2)

        self.feature_predictor = conv2d(512, 512, 5, 1, 2)
        self.occlusion_predictor = conv2d(512, 1, 5, 1, 2)

        bottleneck = []
        for i in range(num_residual_mod_blocks):
            bottleneck.append(res_mod_block2d(512, 8 + 256))
        self.bottleneck = nn.ModuleList(bottleneck)

        self.decoder = nn.Sequential(
            up_block2d(512, 256),
            up_block2d(256, 128),
            up_block2d(128, 64),
            conv2d(64, 3, 7, 1, 3)
        )

        full_grid = make_coordinate_grid(256, 256)
        homo_full_grid = torch.cat([full_grid, torch.ones(256, 256, 1)], dim=2)
        self.register_buffer('homo_full_grid', homo_full_grid) # (256, 256, 3)

        grid = make_coordinate_grid(32, 32)
        homo_grid = torch.cat([grid, torch.ones(32, 32, 1)], dim=2)
        self.register_buffer('grid', grid) # (h, w, 2)
        self.register_buffer('homo_grid', homo_grid) # (h, w, 3)


    def forward(self, src, src_head, drv_head, drv_eyes, drv_audio):
        affine = torch.matmul(src_head['affine'], torch.inverse(drv_head['affine'])) # (n, c, 3, 3)
        affine = affine * torch.sign(affine[:,:,0:1,0:1]) # revert_axis_swap
        affine = affine[:,:,None,None] # (n, c, 1, 1, 3, 3)
        
        affine_motion = torch.matmul(affine, self.homo_full_grid[...,None]) # (n, c, h, w, 3, 1)
        affine_motion = affine_motion[...,:2,0] # (n, c, h, w, 2)
        n, c, h, w, _ = affine_motion.shape

        stacked_src = src.repeat(c, 1, 1, 1)
        flatten_affine_motion = affine_motion.view(n*c, h, w, 2)
        transformed_src = F.grid_sample(stacked_src, flatten_affine_motion, align_corners=False)
        transformed_src = transformed_src.view(n, c*3, h, w)

        # encoding source and tansformed source
        stacked_input = torch.cat([src, transformed_src], dim=1)
        if self.using_gaussian:
            gaussian = drv_head['gaussian'] - src_head['gaussian']
            gaussian = F.interpolate(gaussian, scale_factor=4)
            stacked_input = torch.cat([stacked_input, gaussian], dim=1)
        x = self.encoder(stacked_input)

        # compute dense motion
        mask = F.softmax(self.mask_predictor(x), dim=1) # (n, c+1, h, w)
        dense_motion = torch.matmul(affine, self.homo_grid[...,None]) # (n, c, h, w, 3, 1)
        dense_motion = dense_motion[...,:2,0] # (n, c, h, w, 2)
        identity_motion = self.grid[None, None].repeat(n, 1, 1, 1, 1) # (n, 1, h, w, 2)
        dense_motion = torch.cat([dense_motion, identity_motion], dim=1) # (n, c+1, h, w, 2)
        optical_flow = (mask[...,None] * dense_motion).sum(dim=1) # (n, h, w, 2)

        # compute deformed source feature
        feature = self.feature_predictor(x) # (n, c, h, w)
        deformed = F.grid_sample(feature, optical_flow, align_corners=False) # (n, c, h, w)
        occlusion = torch.sigmoid(self.occlusion_predictor(x)) # (n, 1, h, w)
        feature = occlusion * deformed # (n, c, h, w)

        # inject local motion
        local_motion = torch.cat([drv_eyes, drv_audio], dim=1)
        for i in range(len(self.bottleneck)):
            feature = self.bottleneck[i](feature, local_motion)

        prediction = torch.sigmoid(self.decoder(feature))

        outputs = {
            'prediction': prediction, # (n, 3, h, w)
            'transformed_src': transformed_src, # (n, c*3, h, w)
            'mask': mask, # (n, c+1, h, w)
            'occlusion': occlusion, # (n, 1, h, w)
            'optical_flow': optical_flow # (n, h, w, 2)
        }
        return outputs
