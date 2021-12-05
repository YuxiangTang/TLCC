import torch.nn as nn
import math
import torch
from . import SqueezeNet_ada, rgb2uvHist, ColorChannelAttention, DeviceChannelAttention

class TLCC(nn.Module):
    def __init__(self, normalization):
        super(TLCC, self).__init__()
        # add uvHist
        self.hist2ccm = rgb2uvHist()

        # backbone
        self.squeezenet = SqueezeNet_ada(3, normalization=normalization)
        # self.dca = DeviceChannelAttention(512, 512)
        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(320 * 2, 64, kernel_size=6, stride=1, padding=3),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.5),
                  nn.Conv2d(64, 3, kernel_size=1, stride=1)
        )
        self.init_weight()
        self.eps = torch.tensor(1e-9)
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def MatrixNormalizationLayer(self, mat, abs_mode=False):
        if abs_mode:
            mat = torch.abs(mat)
        mat_norm = mat / (torch.sum(mat, 3, keepdim=True) + self.eps) + self.eps
        return mat_norm

    def MatrixInverseLayer(self, mat):
        ccm_inv = torch.linalg.inv(mat)
        return ccm_inv

    def img_convert(self, img, ccm):
        img = img.permute(0, 2, 3, 1)
        img = torch.matmul(img, ccm.transpose(3, 2))
        img = img.permute(0, 3, 1, 2)
        return img
    
    def ill_convert(self, ill, ccm):
        ccm_inv = self.MatrixInverseLayer(ccm)
        return torch.matmul(ill.unsqueeze(dim=1).unsqueeze(dim=1), ccm_inv.transpose(3, 2)).squeeze(dim=1).squeeze(dim=1)
    

    def forward(self, x):
        ccm, device_feature = self.hist2ccm(x)
        ccm = self.MatrixNormalizationLayer(ccm, abs_mode=True)
        x = self.img_convert(x, ccm)
        x = self.squeezenet(x, device_feature)
        #x = self.dca(x)
        x = self.fc(x)
        ill = nn.functional.normalize(torch.sum(x,dim=(2,3)), dim=1)
        ill = self.ill_convert(ill, ccm)
        return ill
