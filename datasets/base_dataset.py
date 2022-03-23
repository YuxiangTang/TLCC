"""
Some operations about SE-Scheme and other basic operations.
"""

import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import math
from tools import *

class base_dataset(Dataset):
    def __init__(self, data_dir, minik, input_size, mode):
        self.data_dir = data_dir
        self.minik = 1 # minik
        self.input_size = input_size
        self.mode = mode  

    def mapping_into_statis(self, img):
        """
        Core operation that maps the illuminant into statistic form.
        """
        stat = self.statis_norm(img) 
        remove = img * stat * math.sqrt(3)
        return remove, stat
    
    def statis_norm(self, img):
        """
        Core operation that calculate the h(I, n, sigma, p)^-1
        """
        # compute h(~)
        val = np.mean(np.power(img, self.minik), (0, 1)) 
        stat = np.power(val, 1 / self.minik) 
        stat = self.L2_norm(stat) 
        
        # compute h(~)^-1 
        stat = 1 / (stat + 1e-20)
        
        stat = self.L2_norm(stat) 
        return stat 

    def resize(self, img):
        return cv2.resize(img, (self.input_size, self.input_size))
    
    def L2_norm(self, vec):
        return vec / (np.linalg.norm(vec, 2) + 1e-20)
    
    def load_nameseq(self, dir_path):
        img_list = []
        with open(dir_path, "r") as f:
            for line in f:  
                line = line.rstrip()
                img_list.append(line)
        return img_list
