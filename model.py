import torch.nn as nn

from modules.utils import AntiAliasInterpolation2d
from modules.head_predictor import HeadPredictor
from modules.eyes_encoder import EyesEncoder
from modules.generator import Generator


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.down_sampler = AntiAliasInterpolation2d(3, 0.25) # 1/4 resolution
        self.head_predictor = HeadPredictor(**params['head_predictor'])
        self.eyes_encoder = EyesEncoder()
        self.audio_encoder = params['audio_encoder']()
        self.generator = Generator(**params['generator'])


    def forward(self, src, drv, eye, spec):
        src_down = self.down_sampler(src)
        drv_down = self.down_sampler(drv)

        src_head = self.head_predictor(src_down)
        drv_head = self.head_predictor(drv_down)

        drv_eyes = self.eyes_encoder(eye)

        drv_audio = self.audio_encoder(spec)

        generator_out = self.generator(src, src_head, drv_head, drv_eyes, drv_audio)

        return generator_out['prediction']
