from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from scipy.io import loadmat
from .base_dataset import base_dataset
import torch
import math
import pickle
import random
from tools import *

class ILL(base_dataset):
    def __init__(self, dataset_name, data_dir, mode, fold_idx, minik = 1, input_size = 512):
        base_dataset.__init__(self, data_dir, minik, input_size, mode)
        self.dataset_name = dataset_name
        self.img_list = self.three_fold(fold_idx)
        
        self.ANGLE = 60
        self.SCALE = [0.2, 1.0]
        self.AUG_NUM = 4
        self.AUG_COLOR = 0.8
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img_path = self.data_dir + self.img_list[idx]
        img = np.load(img_path + '.npy').astype(np.float32)
        mask = np.load(img_path + '_mask.npy').astype(np.bool)
        illums = np.load(img_path + '_gt.npy').astype(np.float32)
        camera = str(np.load(img_path + '_camera.npy'))
        img = img / np.max(img) * 65535.0
        idx1, idx2, _ = np.where(mask == False)
        img[idx1, idx2, :] = 1e-5

        if self.mode == 'train':
            img_batch = []
            ill_batch = []
            for i in range(self.AUG_NUM):
                img_aug, ill_aug = self.augment_train(img, illums)
                img_batch.append(img_aug)
                ill_batch.append(ill_aug)
            img = np.stack(img_batch)
            ill = np.stack(ill_batch)
            img = img[:,:,:,::-1] / 65535
            img = np.power(img,(1.0/2.2))
            img = img.transpose(0,3,1,2)
        else:
            ill = illums
            img = img[:,:,::-1] / 65535
            img = np.power(img,(1.0/2.2)) 
            img = img.transpose(2,0,1)
        img = torch.from_numpy(img.copy()) 
        ill = torch.from_numpy(ill.copy())
        # img /= np.max(img)
        # remove_stat, stat, camera_onehot = self.make_input(img, ill, 'Canon1D')
        return img, ill, camera 

    def three_fold(self, idx):
        img_list = []
        if self.mode == 'train':
            if self.dataset_name == 'all':
                dataset_lst = ['NUS_full', 'CC_full']
            else:
                dataset_lst = [self.dataset_name + '_full']
            for ds in dataset_lst:
                for i in range(3):
                    if i == idx:
                        continue
                    img_list += self.load_nameseq(self.data_dir + '/{}_fold{}.txt'.format(ds, i))
        else:
            img_list += self.load_nameseq(self.data_dir + '/{}_full_fold{}.txt'.format(self.dataset_name, idx))
        return img_list
        
    def augment_train(self,ldr, illum):
        angle = (random.random() - 0.5) * self.ANGLE
        scale = math.exp(random.random() * math.log(self.SCALE[1] / self.SCALE[0])) * self.SCALE[0]
        s = int(round(min(ldr.shape[:2]) * scale))
        s = min(max(s, 10), min(ldr.shape[:2]))
        start_x = random.randrange(0, ldr.shape[0] - s + 1)
        start_y = random.randrange(0, ldr.shape[1] - s + 1)        
        flip_lr = random.randint(0, 1) # Left-right flip?   
        color_aug = np.zeros(shape=(3, 3))
        for i in range(3):
            color_aug[i, i] = 1 + random.random() * self.AUG_COLOR - 0.5 * self.AUG_COLOR 
        
        def crop(img, illumination):
            if img is None:
                return None
            img = img[start_x:start_x + s, start_y:start_y + s]
            img = rotate_and_crop(img, angle)
            img = cv2.resize(img, (self.input_size, self.input_size))
            if flip_lr:
                img = img[:, ::-1]
            img = img.astype(np.float32)
            new_illum = np.zeros_like(illumination)
            # RGB -> BGR
            illumination = illumination[::-1]
            for i in range(3):
                for j in range(3):
                    new_illum[i] += illumination[j] * color_aug[i, j]

            img *= np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],dtype=np.float32)
            new_image = img
            new_image = np.clip(new_image, 0, 65535)
            new_illum = np.clip(new_illum, 0.01, 100)        
            return new_image, new_illum[::-1]            
        return crop(ldr, illum)
