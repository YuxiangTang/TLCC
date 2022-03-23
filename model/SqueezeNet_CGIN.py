"""
We improved on the basis of SqueezeNet v1.1 and added 
the proposed Color-Guided Instance Normalization module (CGIN).
"""

import torch.nn as nn
import math
import torch


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CGIN(nn.Module):
    """
    Our proposed Color-Guided Instance Normalization (CGIN) module.
    
    We drive the color feature to adaptively regularize the feature map.
    
    :param channels: set the channels same as the feature map.
    
    Return: A Tensor, the size same as input.
    """
    def __init__(self, channels, epsilon=1e-9):
        super(CGIN, self).__init__()
        self.eps = epsilon
        self.squeeze_channels = int(channels)  
        self.conv = nn.Sequential(
            nn.Conv2d(512, self.squeeze_channels, kernel_size=3, stride=2, bias=True, padding=0),  
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Dropout(p=0.5),
        )
        # fully convoltion: the shape of output tensor is (bn, c, 1, 1)
        self.gamma_fc = nn.Sequential(
            nn.Conv2d(self.squeeze_channels, channels, kernel_size=3, stride=1, bias=True, padding=0),
        )
        self.beta_fc = nn.Sequential(
            nn.Conv2d(self.squeeze_channels, channels, kernel_size=3, stride=1, bias=True, padding=0), 
        )

    def forward(self, x, device_feature):
        bn, c, h, w = x.shape
        
        mu = x.mean(dim=(2, 3), keepdim=True)
        sigma = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        x_norm = (x - mu) / torch.sqrt(sigma + self.eps)
        
        tmp = self.conv(device_feature)
        gamma = self.gamma_fc(tmp)  
        beta = self.beta_fc(tmp)  
        return x_norm * gamma + beta


class AdaFire(nn.Module):
    """
    We maintain four versions of Fireblock.
    
    when normalization
        is 'CGIN': adds our proposed CGIN module.
        is 'CGIN_squ': a light version of TLCC that adds CGIN module in squeeze layer only.
        is 'IN': adds instance normlization.
        is 'None': nothing adds. Same as the original FireBlock.
    
    :param in_planes: in_planes of the input feature
    :param squeeze_planes: the squeeze size of the squeeze operation
    :param expand_planes: the output size after squeeze operation
    
    Return: A Tensor with size --> bn * (expand_planes * 2) * H * W
            Feature size will not change.
    """
    def __init__(
        self,
        in_planes,
        squeeze_planes,
        expand_planes,
        use_shortcut=True,
        normalization='CGIN'
    ):
        super(AdaFire, self).__init__()
        assert normalization in ['CGIN', 'CGIN_squ', 'IN', 'None']
        self.normalization = normalization
        self.use_shortcut = use_shortcut

        self.squeeze = nn.Conv2d(in_planes, squeeze_planes, kernel_size=1, stride=1)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.expend11 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.expend33 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(inplace=True)

        if normalization == 'CGIN':
            self.din1 = CGIN(squeeze_planes)
            self.din2 = CGIN(expand_planes)
            self.din3 = CGIN(expand_planes)
        elif normalization == 'CGIN_squ':
            self.din1 = CGIN(squeeze_planes)
            self.din2 = Identity()
            self.din3 = Identity()
        elif normalization == 'IN':
            self.din1 = nn.InstanceNorm2d(squeeze_planes)
            self.din2 = nn.InstanceNorm2d(expand_planes)
            self.din3 = nn.InstanceNorm2d(expand_planes)
        else:
            self.din1 = Identity()
            self.din2 = Identity()
            self.din3 = Identity()

        if self.use_shortcut:
            if in_planes != expand_planes * 2:
                self.downsample = nn.Conv2d(in_planes, expand_planes * 2, kernel_size=1, stride=1)
            else:
                self.downsample = Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_in, device_feature):
        x = self.squeeze(x_in)
        x = self.din1(x) if self.normalization in ['IN', 'None'] else self.din1(x, device_feature)
        x = self.relu1(x)

        out1 = self.expend11(x)
        out1 = self.din2(out1) if self.normalization in ['IN', 'None'] else self.din2(out1, device_feature)
        out1 = self.relu2(out1)

        out2 = self.expend33(x)
        out2 = self.din3(out2) if self.normalization in ['IN', 'None'] else self.din3(out2, device_feature)
        out2 = self.relu3(out2)

        out = torch.cat([out1, out2], 1)

        if self.use_shortcut:
            out = out + self.downsample(x_in)

        return out


class fire(nn.Module):
    """
    Classic FireBlock
    """
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.expend11 = nn.Sequential(
            nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.expend33 = nn.Sequential(
            nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.squeeze(x)
        out1 = self.expend11(x)
        out2 = self.expend33(x)
        out = torch.cat([out1, out2], 1)
        return out


class SqueezeNet_ada(nn.Module):  # v1.1
    """
    We replace the FireBlock with AdaFireBlock (added proposed CGIN).
    """
    def __init__(self, in_channel, normalization):
        super(SqueezeNet_ada, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=2)  # H/2*W/2
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # H/4*W/4
        self.fire2 = AdaFire(64, 16, 64, normalization=normalization)
        self.fire3 = AdaFire(128, 16, 64, normalization=normalization)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # H/8*W/8
        self.fire4 = AdaFire(128, 32, 128, normalization=normalization)
        self.fire5 = AdaFire(256, 32, 128, normalization=normalization)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # H/16*W/16
        self.fire6 = AdaFire(256, 48, 192, normalization=normalization)
        self.fire7 = AdaFire(384, 48, 192, normalization=normalization)
 
        self.fire8 = AdaFire(384, 64, 256, normalization=normalization)
        # self.fire9 = AdaFire(512, 64, 256, normalization=normalization)
        # self.fire10 = AdaFire(512, 64, 256, normalization=normalization)
        # self.fire11 = AdaFire(512, 64, 256, normalization=normalization)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, device_feature):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x, device_feature)
        x = self.fire3(x, device_feature)
        x = self.maxpool2(x)
        x = self.fire4(x, device_feature)
        x = self.fire5(x, device_feature)
        x = self.maxpool3(x)
        x = self.fire6(x, device_feature)
        x = self.fire7(x, device_feature)

        x = self.fire8(x, device_feature)
        # x = self.fire9(x, device_feature)
        # x = self.fire10(x, device_feature)
        # x = self.fire11(x, device_feature)
        return x
