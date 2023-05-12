import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.transforms import functional as TF
import numpy as np
class RepB(nn.Module):


    def __init__(self, n_feats):
        super(RepB, self).__init__()
        self.conv1_1 = nn.Conv2d(n_feats,n_feats,3,1,1)
        self.conv1_2 = nn.Conv2d(n_feats,n_feats,1)

        self.conv2 = nn.Conv2d(n_feats,n_feats,1)

    def forward(self, x):
        out1 = self.conv1_2(self.conv1_1(x))
        out2 = self.conv2(x)

        return out1+out2
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0*g:1*g, 1, 2] = 1.0
        mask[:, 1*g:2*g, 1, 0] = 1.0
        mask[:, 2*g:3*g, 2, 1] = 1.0
        mask[:, 3*g:4*g, 0, 1] = 1.0
        mask[:, 4*g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1) 
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0*g:1*g, 0, 1, 2] = 1.0 ## left
        self.weight[1*g:2*g, 0, 1, 0] = 1.0 ## right
        self.weight[2*g:3*g, 0, 2, 1] = 1.0 ## up
        self.weight[3*g:4*g, 0, 0, 1] = 1.0 ## down
        self.weight[4*g:, 0, 1, 1] = 1.0 ## identity     

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y) 
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory': 
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y

class LFE(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='gelu'):
        super(LFE, self).__init__()    
        self.exp_ratio = exp_ratio
        self.act_type  = act_type

        # self.conv0 = RepB(inp_channels)
        # self.conv1 = RepB(out_channels)
        # self.conv0 = nn.Conv2d(inp_channels, out_channels*exp_ratio,3,1,1)
        # self.conv1 = nn.Conv2d(out_channels*exp_ratio, out_channels,1)
        self.conv0 = ShiftConv2d(inp_channels, out_channels*exp_ratio)
        self.conv1 = ShiftConv2d(out_channels*exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y_ = self.act(y)
        y = self.conv1(y_) 
        return y
# mask_1 = torch.tensor([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],dtype=bool)
# mask_2 = torch.tensor([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],dtype=bool)                   

class WSA(nn.Module):

    def __init__(self, dim, window_size,  qkv_bias=True, qk_scale=None, shifts=0, attn_drop=0., proj_drop=0.,calc_attn=True):

        super().__init__()
        self.dim = dim
        self.shifts = shifts
        self.window_size = window_size  # Wh, Ww
        self.project_inp = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1), 
            nn.BatchNorm2d(self.dim)
        )
        self.project_out= nn.Conv2d(dim, dim, kernel_size=1)
        self.padding_mode = 'seq_refl_win_pad'
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask = nn.Conv2d(self.dim, self.dim, kernel_size=1,stride=2)

    def seq_refl_win_pad(self, x, back=False):
        if self.window_size == 1: return x
        x = TF.pad(x, (0,0,self.window_size,self.window_size)) if not back else TF.pad(x, (self.window_size,self.window_size,0,0))
        #import pdb;pdb.set_trace()
        if self.padding_mode == 'zero_pad':
            return x
        if not back:
            (start_h, start_w), (end_h, end_w) = to_2tuple(-2*self.window_size), to_2tuple(-self.window_size)
            # pad lower
            x[:,:,-(self.window_size):,:] = x[:,:,start_h:end_h,:].contiguous()
            # pad right
            x[:,:,:,-(self.window_size):] = x[:,:,:,start_w:end_w].contiguous()
        else:
            (start_h, start_w), (end_h, end_w) = to_2tuple(self.window_size), to_2tuple(2*self.window_size)
            # pad upper
            x[:,:,:self.window_size,:] = x[:,:,start_h:end_h,:].contiguous()
            # pad left
            x[:,:,:,:self.window_size] = x[:,:,:,start_w:end_w].contiguous()
            
        return x

    def forward(self, x):
        b,c,h,w = x.shape
        if self.shifts > 0:
            x = torch.roll(x, shifts=(-self.window_size//2, -self.window_size//2), dims=(2,3))

        x = self.project_inp(x) # b, c, h, w

        x_pad = self.seq_refl_win_pad(x, False)
        gla_X = x_pad.unfold(3, 2*self.window_size, self.window_size).unfold(2, 2*self.window_size, self.window_size)
        gla_X = rearrange(
            gla_X, 'b c hh hw h w -> (b hh hw) c h w', 
            h=self.window_size*2, w=self.window_size*2
        )
        gla_X = self.mask(gla_X)
        q = rearrange(
            gla_X, 'bn c  h w -> bn (h w) c', 
        )

        v = rearrange(
            x, 'b c (h dh) (w dw) -> (b h w) (dh dw) c', 
            dh=self.window_size, dw=self.window_size
        )
        # q = q + gla_X
        atn = (q @ q.transpose(-2, -1)) 
        #import pdb;pdb.set_trace()
        atn = atn.softmax(dim=-1)
        y = (atn @ v)
        y = rearrange(
            y, '(b h w) (dh dw) c-> b ( c) (h dh) (w dw)', 
            h=h//self.window_size, w=w//self.window_size, dh=self.window_size, dw=self.window_size
        )
        if self.shifts > 0:
            y = torch.roll(y, shifts=(self.window_size//2, self.window_size//2), dims=(2, 3))
        y = self.project_out(y) # b, c, h, w
        return y

class ELAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_size= 8, shared_depth=1):
        super(ELAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_size = window_size
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth
        
        modules_lfe = {}
        modules_gmsa = {}
        modules_lfe['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa['gmsa_0'] = WSA(dim=inp_channels, shifts=shifts, window_size=window_size, calc_attn=True)
        for i in range(shared_depth):
            modules_lfe['lfe_{}'.format(i+1)] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
            modules_gmsa['gmsa_{}'.format(i+1)] = WSA(dim=inp_channels, shifts=shifts, window_size=window_size, calc_attn=False)
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)

    def forward(self, x):
        for i in range(1 + self.shared_depth):
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y = self.modules_gmsa['gmsa_{}'.format(i)](x)
                x = y + x
        return x
