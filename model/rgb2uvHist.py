import torch
import torch.nn as nn
import torch.nn.functional as F
from .SqueezeNet_adain import fire

class rgb2uvHist(nn.Module):
    def __init__(self, hist_num = 60):
        super(rgb2uvHist, self).__init__()
        self.hist_num = hist_num + 1
        self.eps = 6.4 / hist_num
        self.size = 150
        self.A = torch.nn.Parameter(torch.repeat_interleave(torch.unsqueeze(torch.arange(-3.2, 3.21, self.eps), dim=0), 
                                                            self.size * self.size, dim=0).view((1, 1, -1, self.hist_num)),
                                    requires_grad = False)
        self.Asize = torch.nn.Parameter(torch.tensor(self.A.shape[2]), requires_grad = False)
        self.sigma_u = nn.Parameter(torch.ones(3).view((1, 3, 1, 1)).float())
        self.sigma_v = nn.Parameter(torch.ones(3).view((1, 3, 1, 1)).float())
        self.C = nn.Parameter(torch.ones(3).view((1, 3, 1, 1)).float()) 
        self.lower = torch.tensor(1e-9)
        
        # sensor-net
        '''
        self.conv = nn.Sequential(                 
            nn.Conv2d(3, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 512, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
        )
        '''
        
        # sensor-net
        self.conv = nn.Sequential(                 
            nn.Conv2d(3, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 512, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(512, 9, kernel_size=6, stride=1),
        )
    
    def rgb2uvhist_param(self, img):
        bn, c, _, _ = img.shape
        img = F.interpolate(img, size=(self.size, self.size), mode="bilinear", align_corners=False)
        img_channelwise = img.view((bn, 3, -1))
        Iy = torch.norm(img_channelwise, dim=1, keepdim=True).reshape(bn, 1, 1, -1)
        father_idx = torch.tensor([0,1,2])
        son1_idx   = torch.tensor([1,0,0])
        son2_idx   = torch.tensor([2,2,1])
        father = img_channelwise[:, father_idx, :]
        son1 = img_channelwise[:, son1_idx, :]
        son2 = img_channelwise[:, son2_idx, :]
        Iu = torch.log(father / son1).view((bn, 3, -1, 1))
        Iv = torch.log(father / son2).view((bn, 3, -1, 1))
        diff_u = torch.pow(torch.abs(Iu - self.A), 2)
        diff_v = torch.pow(torch.abs(Iv - self.A), 2)
        diff_u = torch.exp( - diff_u / (torch.pow(self.sigma_u, 2) + self.lower))
        diff_v = torch.exp( - diff_v / (torch.pow(self.sigma_v, 2) + self.lower))
        tu = diff_u.permute(0, 1, 3, 2)
        tv = diff_v.permute(0, 1, 2, 3)
        hist = torch.matmul(Iy * tu, tv)
        hist_norm = torch.sqrt(hist * self.C)
        return hist_norm

    def forward(self, x):
        x = self.rgb2uvhist_param(x)
        feature = self.conv(x)
        # print(feature.shape)
        # x_out = feature.view(bn, -1)
        x_out = self.fc(feature).view(feature.shape[0], 1, 3, 3)
        # x_out = self.MatrixNormalizationLayer(x_out)
        return x_out, feature# .clone().detach()
