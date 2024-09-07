from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.Multiscale_Hybrid_Attention as MHA
import src.Multiscale_Hybrid_Residual_Conv as MHRC
#from torchstat import stat

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ConvRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvRelu, self).__init__(
            MHRC.MHRC(inc=in_channels, outc=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class AttentionRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(AttentionRelu, self).__init__(
            MHA.TP_Block(image_size=512, patch_size=16, inc=in_channels, outc=out_channels),
            nn.ReLU(inplace=True)
        )


class Down1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down1, self).__init__(
            ConvRelu(in_channels, out_channels)
        )

class Down2(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down2, self).__init__(
            AttentionRelu(in_channels, out_channels)
            #DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.conv2 = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)

        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class TP_UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_c: int = 64):
        super(TP_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.in_conv1 = ConvRelu(in_channels, base_c)
        self.down1 = Down2(base_c, base_c*2)
        self.down2 = Down2(base_c*2, base_c*4)
        self.down3 = Down2(base_c*4, base_c*8)

        self.up1 = Up(base_c*8, base_c*4)
        self.up2 = Up(base_c*4, base_c*2)
        self.up3 = Up(base_c*2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv1(x)
        #print('x1', x1.shape)
        x2 = self.down1(x1)
        #print('x2', x2.shape)
        x3 = self.down2(x2)
        #print('x3', x3.shape)
        x4 = self.down3(x3)
        #print('x4', x4.shape)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.out_conv(x)

        return {'out': logits}


#model = TP_UNet(in_channels=3, base_c=16)
#print(model)
#x = torch.randn(1,3,256,256)
#pre = model(x)
#print(pre.shape)
#stat(model, (3,224,224))
