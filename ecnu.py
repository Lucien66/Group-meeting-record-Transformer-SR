# -----------------------------------------------------------------------------------
# Swin2SR: Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration, https://arxiv.org/abs/2209.11345
# Written by Conde and Choi et al.
# -----------------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import models.modules.block as block




def ecnu(scale=2):
    """
    Define SRModel architecture here and return instance.
    """
    model = ECNU(in_channels=3, out_channels=3, upscale=scale)
    return model

#------------------------------------------------train---------------------------------------------------------------------
#block=3,channel=32,time=27.9,block=2,channel=48,time=34,block=2,channel=32,conv=1,time=18
class ECNU(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN_S in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=64,
                 upscale=2):
        super(ECNU, self).__init__()

        self.conv_1 = block.conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)
        self.down_blocks = 8
        backbone_expand_ratio = 2
        attention_expand_ratio = 2
        ERBs = [ERB(feature_channels, backbone_expand_ratio) for _ in range(self.down_blocks)]
        HFABs = [HFAB(feature_channels, 1, feature_channels//2, attention_expand_ratio) for i in range(self.down_blocks)]
        self.ERBs = nn.ModuleList(ERBs)
        self.HFABs = nn.ModuleList(HFABs)

        #self.esa = HFAB(feature_channels ,1, 16,2)

        # self.conv_2 = block.conv_layer(feature_channels,
        #                                feature_channels,
        #                                kernel_size=3)

        self.upsampler = block.pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)
        h = out_feature
        for i in range(self.down_blocks):
            h = self.ERBs[i](h)
            h = self.HFABs[i](h)
        #out_b4 = self.esa(h)


        out_low_resolution = h + out_feature
        input_feature = nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        output = self.upsampler(out_low_resolution) + input_feature

        return output
lrelu_value = 0.1
act = nn.LeakyReLU(lrelu_value)
def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t


def get_bn_bias(bn_layer):
    gamma, beta, mean, var, eps = bn_layer.weight, bn_layer.bias, bn_layer.running_mean, bn_layer.running_var, bn_layer.eps
    std = (var + eps).sqrt()
    bn_bias = beta - mean * gamma / std

    return bn_bias


class RRRB(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.
    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(RRRB, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, ratio*n_feats, 1, 1, 0)
        self.fea_conv = nn.Conv2d(ratio*n_feats, ratio*n_feats, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(ratio*n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return out


class ERB(nn.Module):
    """ Enhanced residual block for building FEMN.
    Diagram:
        --RRRB--LeakyReLU--RRRB--
        
    Args:
        n_feats (int): Number of feature maps.
        ratio (int): Expand ratio in RRRB.
    """

    def __init__(self, n_feats, ratio=2):
        super(ERB, self).__init__()
        self.conv1 = RRRB(n_feats, ratio)
        self.conv2 = RRRB(n_feats, ratio)

    def forward(self, x):
        out = self.conv1(x)
        out = act(out)
        out = self.conv2(out)

        return out


class HFAB(nn.Module):
    """ High-Frequency Attention Block.
    Diagram:
        ---BN--Conv--[ERB]*up_blocks--BN--Conv--BN--Sigmoid--*--
         |___________________________________________________|
    Args:
        n_feats (int): Number of HFAB input feature maps.
        up_blocks (int): Number of ERBs for feature extraction in this HFAB.
        mid_feats (int): Number of feature maps in ERB.
    Note:
        Batch Normalization (BN) is adopted to introduce global contexts and achieve sigmoid unsaturated area.
    """

    def __init__(self, n_feats, up_blocks, mid_feats, ratio):
        super(HFAB, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_feats)
        self.bn2 = nn.BatchNorm2d(mid_feats)
        self.bn3 = nn.BatchNorm2d(n_feats)

        self.squeeze = nn.Conv2d(n_feats, mid_feats, 3, 1, 0)

        convs = [ERB(mid_feats, ratio) for _ in range(up_blocks)]
        self.convs = nn.Sequential(*convs)

        self.excitate = nn.Conv2d(mid_feats, n_feats, 3, 1, 0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # explicitly padding with bn bias
        out = self.bn1(x)
        bn1_bias = get_bn_bias(self.bn1)
        out = pad_tensor(out, bn1_bias) 

        out = act(self.squeeze(out))
        out = act(self.convs(out))

        # explicitly padding with bn bias
        out = self.bn2(out)
        bn2_bias = get_bn_bias(self.bn2)
        out = pad_tensor(out, bn2_bias)

        out = self.excitate(out)

        out = self.sigmoid(self.bn3(out))

        return out * x
#------------------------------------------------test---------------------------------------------------------------------
class ECNU(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN_S in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=64,
                 upscale=2):
        super(ECNU, self).__init__()

        self.conv_1 = block.conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)
        self.down_blocks = 8
        backbone_expand_ratio = 2
        ERBs = [ERB(feature_channels) for _ in range(self.down_blocks)]
        #---
        HFABs  = [HFAB(feature_channels, 1, feature_channels//2) for i in range(self.down_blocks)]
        self.ERBs = nn.ModuleList(ERBs)
        #---
        self.HFABs = nn.ModuleList(HFABs)

        #self.esa = HFAB(feature_channels ,1, 16)

        # self.conv_2 = block.conv_layer(feature_channels,
        #                                feature_channels,
        #                                kernel_size=3)

        self.upsampler = block.pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)
        h = out_feature
        for i in range(self.down_blocks):
            h = self.ERBs[i](h)
            #---
            h = self.HFABs[i](h)
       # out_b4 = self.esa(h)


        out_low_resolution = h + out_feature
        input_feature = nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        output = self.upsampler(out_low_resolution) + input_feature

        return output

lrelu_value = 0.1
act = nn.LeakyReLU(lrelu_value)
#nn.PReLU(num_parameters=1, init=0.05)
class RRRB(nn.Module):
    def __init__(self, n_feats):
        super(RRRB, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)

        return out
class SparseConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, sparse_act=None):
        super(SparseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.sparse_act = sparse_act
        self.is_searched = False
        self.zeta = nn.Parameter(torch.ones(out_channels, dtype=torch.float))  # channel-wise
        self.searched_zeta = torch.ones_like(self.zeta)

    def forward(self, input):
        z = self.searched_zeta if self.is_searched else self.zeta
        z = z.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        out = out * z
        return out

    def get_zeta(self):
        # return self.searched_zeta if self.is_searched else self.zeta
        return self.zeta

    def cal_zeta_threshold(self, cut_channel=32, is_abs=True):
        zeta = self.get_zeta().clone()
        if is_abs:
            zeta = torch.abs(zeta)
        zeta = sorted(zeta)
        threshold = zeta[cut_channel-1]
        return threshold

    def compress(self, threshold, is_abs=True):
        self.is_searched = True
        _zeta = self.get_zeta()
        if is_abs:
            _zeta = torch.abs(_zeta)
        _mask = (_zeta > threshold).float()
        self.searched_zeta = self.get_zeta().detach() * _mask
        self.zeta.requires_grad = False
        # print('searched zeta: ', self.searched_zeta)
        print(' ==> [{}/{}] channels deleted: '.format(torch.sum(_mask), self.out_channels))
        # else:
        #     # update weights
        #     new_weight = self.conv.weight.data.clone() * self.get_zeta()
        #     new_bias = self.conv.bias.data.clone() * self.get_zeta()
        #     self.conv.weight.data = new_weight
        #     self.conv.bias.data = new_bias

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True

    # load a compressed model. This method is useful in multi-stages training scenario.
    def freeze(self):
        self.is_searched = True
        self.zeta.requires_grad = False

    def update_weights(self):
        new_weight = self.conv.weight.data.clone() * self.get_zeta()
        new_bias = self.conv.bias.data.clone() * self.get_zeta()
        self.conv.weight.data = new_weight
        self.conv.bias.data = new_bias

class ERB(nn.Module):
    def __init__(self, n_feats):
        super(ERB, self).__init__()
        self.conv1 = RRRB(n_feats)
        #self.conv2 = RRRB(n_feats)

    def forward(self, x):
        res = self.conv1(x)
        res = act(res)
        #res = self.conv2(res)

        return res


class HFAB(nn.Module):
    def __init__(self, n_feats, up_blocks, mid_feats):
        super(HFAB, self).__init__()
        self.squeeze = nn.Conv2d(n_feats, mid_feats, 3, 1, 1)
        convs = [ERB(mid_feats) for _ in range(up_blocks)]
        self.convs = nn.Sequential(*convs)
        self.excitate = nn.Conv2d(mid_feats, n_feats, 3, 1, 1)
    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = act(self.squeeze(x))
        out = act(self.convs(out))
        out = self.excitate(out)
        out = self.sigmoid(out)
        out *= x

        return out