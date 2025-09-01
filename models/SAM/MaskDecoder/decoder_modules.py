import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch

class LocationAdaptiveLearner(nn.Module):
    """docstring for LocationAdaptiveLearner"""

    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels))

    def forward(self, x):
        # x:side5_w (N, n_class*4, H, W)
        x = self.conv1(x)  # (N, n_class*4, H, W)
        x = self.conv2(x)  # (N, n_class*4, H, W)
        x = self.conv3(x)  # (N, n_class*4, H, W)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3))  # (N, n_class, 4, H, W)
        return x
    
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CBAM(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM, self).__init__()
        #通道注意力机制
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #空间注意力机制
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #通道注意力机制
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1)
        channel_out=channel_out*x
        #空间注意力机制
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out
    
class non_local(nn.Module):
    def __init__(self, in_dim_x=1, in_dim_c=1, out_dim_x=1, out_dim_c=1):
        super(non_local, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_c = in_dim_c
        self.out_dim_x = out_dim_x
        self.out_dim_c = out_dim_c

        self.conv_in_q = nn.Conv2d(self.in_dim_x, self.out_dim_x, 1, 1)
        self.conv_in_k = nn.Conv2d(self.in_dim_c, self.out_dim_c, 1, 1)
        self.conv_in_v = nn.Conv2d(self.in_dim_c, self.out_dim_c, 1, 1)
        self.conv_out = nn.Conv2d(self.in_dim_x, self.out_dim_x, 1, 1)

    def forward(self, x, cond, shortcut=True):

        x_cond_q = self.conv_in_q(x)
        x_cond_k = self.conv_in_k(cond)
        x_cond_v = self.conv_in_v(cond)

        atten = F.softmax(x_cond_q * x_cond_k)
        if shortcut:
            out = self.conv_out(atten * x_cond_v) + x
        else:
            out = self.conv_out(atten * x_cond_v)

        return out
    

class BPM(nn.Module):
    def __init__(self, in_dim_x=1, in_dim_c=1, out_dim_x=1, out_dim_c=1):
        super(BPM, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_c = in_dim_c
        self.out_dim_x = out_dim_x
        self.out_dim_c = out_dim_c

        self.conv_in_x = nn.Conv2d(self.in_dim_x, self.out_dim_x, 1, 1)
        self.conv_in_c = nn.Conv2d(self.in_dim_c, self.out_dim_c, 1, 1)
        self.CA = non_local(self.in_dim_x,self.in_dim_x,self.out_dim_x, self.out_dim_x)

    def forward(self, x, cond):
        
        x = self.conv_in_x(x)
        cond = self.conv_in_c(cond)

        x_cond = x + x * F.sigmoid(cond)

        out = self.CA(x_cond, x_cond)

        return out

class TFA(nn.Module):
    def __init__(self, in_dim=1, out_dim=256):
        super(TFA, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        #Suppression stage
        self.CBAM = CBAM(self.in_dim)

        #Filtering stage
        self.filter_13 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=(1,3),padding=(0, 1)),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU()
        )
        self.filter_31 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=(3,1),padding=(1, 0)),
            nn.Sigmoid()
        )
        self.filter_15 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=(1,5),padding=(0, 2)),
            nn.Sigmoid()
        )
        self.filter_51 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=(5,1),padding=(2, 0)),
            nn.Sigmoid()
        )

        #Amplification stage
        self.dc_1 = ASPP_module(self.in_dim, self.in_dim, 1)
        self.dc_2 = ASPP_module(self.in_dim, self.in_dim, 2)
        self.dc_3 = ASPP_module(self.in_dim, self.in_dim, 3)

    def forward(self, x):
        b, c, h, w = x.shape
        #Suppression stage
        x_atten = self.CBAM(x)

        x_13 = self.filter_13(x_atten)
        x_31 = self.filter_31(x_atten)
        x_15 = self.filter_15(x_atten)
        x_51 = self.filter_51(x_atten)

        x_f = x_13 + x_31 +x_15 + x_51

        out = self.dc_3(self.dc_2(self.dc_1(x_f)))

        return out