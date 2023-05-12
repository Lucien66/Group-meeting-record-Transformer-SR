import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.archs.elanblock import ELAB, MeanShift
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class ELAN(nn.Module):
    def __init__(self, scale, colors,window_size,m_elan,c_elan,n_share,r_expand,rgb_range):
        super(ELAN, self).__init__()

        self.scale = scale
        self.colors = colors
        self.window_size = window_size
        self.m_elan  = m_elan
        self.c_elan  = c_elan
        self.n_share = n_share
        self.r_expand = r_expand
        self.img_range = rgb_range
        rgb_mean=(0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_range)
        # self.add_mean = MeanShift(rgb_range, sign=1)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_elan // (1+self.n_share)):
            if (i+1) % 2 == 1: 
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 0, 
                        self.window_size, shared_depth=self.n_share
                    )
                )
            else:              
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 1, 
                        self.window_size, shared_depth=self.n_share
                    )
                )
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        #x = self.sub_mean(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        x = x / self.img_range + self.mean
        #x = self.add_mean(x)
        #import pdb;pdb.set_trace()
        return x[:, :, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_size

        wsize = wsize*self.window_size // math.gcd(wsize, self.window_size)
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    pass