import torch.nn as nn
from . import block
import torch
import torch.nn.functional as F


class RLFN_S(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN_S in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=48,
                 upscale=2):
        super(RLFN_S, self).__init__()

        self.conv_1 = block.conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = block.RLFB(feature_channels)
        self.block_2 = block.RLFB(feature_channels)
        self.block_3 = block.RLFB(feature_channels)
        self.block_4 = block.RLFB(feature_channels)
        self.block_5 = block.RLFB(feature_channels)
        self.block_6 = block.RLFB(feature_channels)

        self.conv_2 = block.conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = block.pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b6) + out_feature
        output = self.upsampler(out_low_resolution)

        return output


class ECNU(nn.Module):


    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=46,
                 mid_channels=48,
                 upscale=2):
        super(ECNU, self).__init__()

        self.conv_1 = block.conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = block.RLFB(feature_channels,mid_channels)
        self.pool = nn.MaxPool2d(2,2)

        self.fre = Fre(feature_channels,feature_channels)
        self.block_high = block.RLFB(feature_channels,mid_channels)

        self.block_2 = block.RLFB(feature_channels,mid_channels)
        self.block_3 = block.RLFB(feature_channels,mid_channels)

        self.up = uplayer(feature_channels,feature_channels,3)
        self.block_4 = block.RLFB(feature_channels*2,mid_channels,feature_channels)


        self.conv_2 = block.conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = block.pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)
        
        out_b1 = self.block_1(out_feature)
        pool_b1 = self.pool(out_b1)
        out_b2 = self.block_2(pool_b1)
        out_b3 = self.block_3(out_b2)
        up_b3 = self.up(out_b3)
        fre_high = self.fre(out_b1) 
        #out_high = self.block_high(fre_high)
        up_b3 = torch.cat([up_b3,fre_high],dim=1)
        out_b4 = self.block_4(up_b3)


        out_low_resolution = self.conv_2(out_b4) + out_feature
        output = self.upsampler(out_low_resolution)

        return output

class Fre(torch.nn.Module):
    def __init__(self,in_ch,out_ch):

        super(Fre, self).__init__()
        self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self,x):
        x_low = self.down(x)
        x_high = x - F.interpolate(x_low, size = x.size()[-2:], mode='bilinear', align_corners=True)

        return x_high

class uplayer(nn.Module):

    def __init__(self,in_ch,out_ch,kernel_size,upsample=2,stride=1,relu=True):
        super(uplayer,self).__init__()
        padding = 3//2
        self.reflection_pad = torch.nn.ReflectionPad2d(padding)
        self.conv = torch.nn.Conv2d(in_ch,out_ch*4,kernel_size,stride)

        if relu:
            self.relu = nn.LeakyReLU(0.2)
        self.up = nn.PixelShuffle(upsample)
    
    def forward(self,x):
        out = self.reflection_pad(x)
        out = self.conv(out)

        if self.relu:
            out = self.relu(out)
        
        out = self.up(out)
        return out