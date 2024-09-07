import torch
import torch.nn as nn
import torch.nn.functional as F
import src.deformconv as df
from dropblock import DropBlock2D
#from torchstat import stat
import cv2 as cv

class SE_block(nn.Module):
    def __init__(self, channel, ratio=1):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class MHRC(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.inc = inc
        self.outc = outc

        #layer 1
        self.branch1 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)

        #layer 2
        #self.branch2_11 = nn.Conv2d(inc, 16, kernel_size=1)
        self.branch2_12 = nn.Conv2d(inc, inc, kernel_size=5, stride=1, padding=2, bias=False, groups=inc)

        #layer 3
        #self.branch3_11 = nn.Conv2d(inc, 16, kernel_size=1)
        self.branch3_12 = nn.Conv2d(inc, inc, kernel_size=7, stride=1, padding=3, bias=False, groups=inc)

        #self.branch4 = df.DeformConv2d(inc, 16, kernel_size=3, padding=1)

        #self.dconv = df.DeformConv2d(inc=24, outc=inc, kernel_size=3, stride=1, padding=1, bias=False)
        self.dconv = nn.Conv2d(3*inc, outc, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.drop = DropBlock2D(block_size=3, drop_prob=0.3)

        self.se = SE_block(channel=inc)
        #self.conv = nn.Conv2d(inc, outc, kernel_size=3, padding=1)

    def forward(self, x):
        y1 = self.branch1(x)
        #y1 = y1*self.se(x)

        #y2 = self.branch2_11(x)
        y2 = self.branch2_12(x)
        #y2 = y2*self.se(x)

        #y3 = self.branch3_11(x)
        y3 = self.branch3_12(x)
        #y3 = y3*self.se(x)

        outputs = [y1, y2, y3]
        y = torch.cat(outputs, dim=1)
        #y = y1+y2+y3
        #outm = self.se(y)*y
        #y = torch.cat([y, x], dim=1)

        outputs1 = self.dconv(y)
        outputs = self.relu(outputs1)
        #outputs = self.drop(outputs)

        return outputs


#x = torch.randn(2,3,256,256)
#model = MHRC(3)
#preds = model(x)
#print(preds.shape)
#stat(model,(3,256,256))
#print(preds.shape)

