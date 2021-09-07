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
        self.minik = minik
        self.input_size = input_size
        self.mode = mode  

    def mapping_into_statis(self, img):
        """
        Core operation that the illuminant into statistic form
        """
        stat = self.statis_norm(img) 
        remove = img * stat * math.sqrt(3)
        return remove, stat
    
    def statis_norm(self, img):
        """
        Core operation that calculate the h(I, n, sigma, p)
        """
        # compute h(~)
        # img = self.gradient(img)
        val = np.mean(np.power(img, self.minik), (0, 1)) 
        stat = np.power(val, 1 / self.minik) 
        stat = self.L2_norm(stat) 
        
        # compute h(~)^-1 
        stat = 1 / (stat + 1e-20)
        stat = self.L2_norm(stat) 
        return stat 
    
    def gradient(self, img):
        # img_blur = cv2.GaussianBlur(img, (7, 7), 1)
        img_blur = img # cv2.gassus(img, (7, 7))
        img_grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)
        img_grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT)
        img_grad = np.sqrt(np.power(img_grad_x, 2) + np.power(img_grad_y, 2))
        # k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        # img_grad = cv2.filter2D(img_blur, -1, kernel=k, borderType=cv2.BORDER_REFLECT) / 9
        return img_grad #img_grad

    def resize(self, img):
        return cv2.resize(img, (self.input_size, self.input_size))
    
    def camera2onehot(self, camera, k):
        return torch.tensor([[int(c==camera) for c in self.camera_mode] for _ in range(k)]).float()

    def generate_mask(self, img, threshold = 250 / 255):
        threshold_num = np.max(img) * threshold
        max_img = np.max(img, 2, keepdims=True)
        mask = np.where(max_img >= threshold_num, False, True)
        return mask
    
    def L2_norm(self, vec):
        return vec / (np.linalg.norm(vec, 2) + 1e-20)
    
    def load_nameseq(self, dir_path):
        img_list = []
        with open(dir_path, "r") as f:
            for line in f:  
                line = line.rstrip()
                img_list.append(line)
        return img_list
    
    
