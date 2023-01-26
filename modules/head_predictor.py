import torch
import torch.nn as nn
import torch.nn.functional as F

from .hourglass import Hourglass
from .layers import conv2d
from .utils import make_coordinate_grid


class HeadPredictor(nn.Module):
    def __init__(self, num_affines, using_scale):
        super(HeadPredictor, self).__init__()
        self.using_scale = using_scale

        self.extractor = Hourglass()
        self.predictor = conv2d(self.extractor.out_filters, num_affines, 7, 1, 3) # linear activation

        self.register_buffer('grid', make_coordinate_grid(64, 64)) # (h, w, 2)
        self.register_buffer('identity', torch.diag(torch.ones(3))) # (3, 3)


    def forward(self, x):
        x = self.predictor(self.extractor(x)) # (n, 1, h, w)

        # convert feature to heatmap
        n, c, h, w = x.shape
        x = x.view(n, c, h*w) # flatten spatially
        heatmap = F.softmax(x, dim=2)
        heatmap = heatmap.view(n, c, h, w) # recover shape: (n, c, h, w)

        # compute statistics of heatmap
        mean = (self.grid * heatmap[...,None]).sum(dim=(2, 3)) # (n, c, 2)
        deviation = self.grid - mean[:,:,None,None] # (n, c, h, w, 2)
        covar = torch.matmul(deviation[...,None], deviation[...,None,:]) # (n, c, h, w, 2, 2)
        covar = (covar * heatmap[...,None,None]).sum(dim=(2, 3)) # (n, c, 2, 2)

        # SVD for extract affine from covariance matrix
        U, S, _ = torch.svd(covar.cpu())
        affine = U.to(covar.device) # rotation matrix: (n, c, 2, 2)
        if self.using_scale:
            S = S.to(covar.device) # (n, c, 2)
            S = torch.diag_embed(S ** 0.5) # scale matrix: (n, c, 2, 2)
            affine = torch.matmul(affine, S) # (n, c, 2, 2)

        # add translation to affine matrix
        affine = torch.cat([affine, mean[...,None]], dim=3) # (n, c, 2, 3)
        homo_affine = self.identity[None].repeat(n, c, 1, 1) # (n, c, 3, 3)
        homo_affine[:,:,:2] = affine # (n, c, 3, 3)

        # convert heatmap to gaussian
        covar_inverse = torch.inverse(covar)[:,:,None,None] # (n, c, 1, 1, 2, 2)
        under_exp = torch.matmul(deviation[...,None,:], covar_inverse) # (n, c, h, w, 1, 2)
        under_exp = torch.matmul(under_exp, deviation[...,None]) # (n, c, h, w, 1, 1)
        under_exp = under_exp[...,0,0] # (n, c, h, w)
        gaussian = torch.exp(-0.5 * under_exp) # (n, c, h, w)

        outputs = {
            'affine': homo_affine, # (n, c, 3, 3)
            'heatmap': heatmap, # (n, c, h, w)
            'gaussian': gaussian # (n, c, h, w)
        }
        return outputs
