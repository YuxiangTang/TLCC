import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import math
from tools import *

class base_dataset(Dataset):
    def __init__(self, data_dir, minik, input_size, mode):
        self.data_dir = data_dir
        self.minik = minik
        self.input_size = input_size
        self.mode = mode
        

    def load_nameseq(self, dir_path):
        img_list = []
        with open(dir_path, "r") as f:
            for line in f:  
                line = line.rstrip()
                img_list.append(line)
        return img_list
    
    def gradient(self, img):
        # img_blur = cv2.GaussianBlur(img, (7, 7), 1)
        img_blur = img # cv2.gassus(img, (7, 7))
        img_grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)
        img_grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT)
        img_grad = np.sqrt(np.power(img_grad_x, 2) + np.power(img_grad_y, 2))
        # k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        # img_grad = cv2.filter2D(img_blur, -1, kernel=k, borderType=cv2.BORDER_REFLECT) / 9
        return img_grad #img_grad
    
    def L2_norm(self, vec):
        return vec / (np.linalg.norm(vec, 2) + 1e-20)

    def statis_norm(self, img):
        # compute h(~)
        # img = self.gradient(img)
        val = np.mean(np.power(img, self.minik), (0, 1)) 
        stat = np.power(val, 1 / self.minik) 
        stat = self.L2_norm(stat) 
        
        # compute h(~)^-1 
        stat = 1 / (stat + 1e-20)
        stat = self.L2_norm(stat) 
        return stat 

    def resize(self, img):
        return cv2.resize(img, (self.input_size, self.input_size))

    def process_LRC(self, LRC, LC, dynamic):
        # LRC /= dynamic
        maxn = 255. / 255
        R = LRC / LC / math.sqrt(3) 
        return R

    def mapping_into_statis(self, img):
        stat = self.statis_norm(img) 
        remove = img * stat * math.sqrt(3)
        return remove, stat
 
    def process_LRC_norm(self, LRC, maxn):
    
        # idx1, idx2, idx3 = np.where(LRC > 0.)
        # LRC[idx1, idx2, idx3] = np.power(LRC[idx1, idx2, idx3], 1 / self.gamma)
        # LRC = np.power(LRC, 1 / self.gamma)
        # LRC, mask = self.random_patch(LRC, mask, 8)
        stat = self.statis_norm(LRC, mask)
        stat = stat / np.linalg.norm(stat, 2) * math.sqrt(3)
        remove = LRC * stat  
        idx1, idx2, idx3 = np.where(mask > 0.)
        remove /= np.max(remove[idx1, idx2, :])
        # remove = np.clip(remove, 0, 1)
        # remove = remove / np.max(remove[idx1, idx2, :]) * maxn
        # remove = remove / np.max(remove)
        return remove, stat
    
    def camera2onehot(self, camera, k):
        return torch.tensor([[int(c==camera) for c in self.camera_mode] for _ in range(k)]).float()

    def generate_mask(self, img, threshold = 250 / 255):
        threshold_num = np.max(img) * threshold
        max_img = np.max(img, 2, keepdims=True)
        mask = np.where(max_img >= threshold_num, False, True)
        return mask
