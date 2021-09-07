import torch
import torch.nn as nn
import math

class Angular_loss(torch.nn.Module):
    def __init__(self):
        super(Angular_loss, self).__init__()
        self.threshold = 0.999999
    
    def forward(self, pred, target):
        return torch.mean(self.angulor_loss(pred, target))
    
    def angulor_loss(self, pred, target):
        # pred = pred.double()
        # target = target.double()
        
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(target, dim=1)

        arccos_num = torch.sum(pred * target, dim=1)
        arccos_num = torch.clamp(arccos_num, -self.threshold, self.threshold)
        angle = torch.acos(arccos_num) * (180 / math.pi)
        return angle
   
    def batch_angular_loss(self, pred, target):
        return self.angulor_loss(pred, target)
