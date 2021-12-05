"""
The dataloader that support CCD, NUS 8, Cube+ and Place205.
When you plan to use this file, you need preprocess the image first.
"""

import numpy as np
import cv2
from .base_dataset import base_dataset
import torch
import math
import random
from tools import *

class MIX(base_dataset):
    def __init__(self, dataset_name, data_dir, mode, fold_idx, minik = 1, multiple=5, input_size = 512, aug_num=4,
     statistic_mode=True, camera_trans=None, bright_occ_mode=False, blur_mode=False):
        """
        :param dataset_name: the name
        :param data_dir: the location of the data and image index file
        :param mode: Train or Valid
        :param fold_idx: set fold_idx as Test fold.
        :param minik: Minkowski norm. to control the statistic.
        :param multiple: Multiply the amount of data in each epoch. 
        :param input_size: the image size
        :param aug_num: due to an image is too large, we augment it multiple 
                        times as we read it once.
        :param statistic_mode: Bool, if True use statistic label, otherwise use illumination label.
        :param camera_trans: inter-camera transformation
        
        :param bright_occ_mode, blur_mode: Abandoned!
        """
        
        
        base_dataset.__init__(self, data_dir, minik, input_size, mode)
        self.dataset_name = dataset_name
        self.camera_trans = camera_trans
        self.statistic_mode = statistic_mode
        self.bright_occ_mode = bright_occ_mode
        self.blur_mode = blur_mode
        self.multiple = multiple
        self.img_list = self.three_fold(fold_idx)
        
        self.camera_mode = ['Canon5D', 'Canon1D', 'Canon550D', 'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', \
            'NikonD5200', 'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57', 'JPG']
            
        self.ANGLE = 60
        self.SCALE = [0.5, 1.0]
        self.AUG_NUM = aug_num
        self.AUG_COLOR = 0.8
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        # combine JPG and RAW
        img_path = self.data_dir + self.img_list[idx]
        dataset = self.img_list[idx].split('/')[1]
        # JPG Preprocess Road
        if dataset in ['Place205', 'Cube_jpg']:  
            if dataset == 'Place205':
                img, camera = self.load_jpg(img_path)
            else:
                img, camera = self.load_cube_jpg(img_path)
            # approximate sRGB image, assmues the illuminant == [1 1 1]
            ill = np.ones((3))
        # RAW Preprocess Road
        else: 
            img, ill, camera = self.load_raw(img_path)    

        img = img * 65535.0 # 0 ~ 1 --> 0 ~ 65535
        img[img == 0] = 1e-5
            
        if self.mode == 'train':
            # For a single image, augmenting multiple times can improve training efficiency 
            img_batch = []
            gt_batch = []
            for i in range(self.AUG_NUM):
                img_aug = self.augment_img(img) # self.transform(image=img)['image']
                remove_stat, stat, si = self.generate_statistic_gt(img_aug, ill) # Convert to SET
                img_aug, gt_aug = self.augment_ill(remove_stat, stat)
                img_aug = img_aug / np.max(img_aug)
                # img_aug = Brightness_Correction(img_aug)
                img_batch.append(img_aug)
                gt_batch.append(gt_aug)
            img = np.stack(img_batch)
            gt = np.stack(gt_batch)
            img = np.power(img, (1.0/2.2))
            img = img.transpose(0, 3, 1, 2)
        else:
            remove_stat, gt, si = self.generate_statistic_gt(img, ill)
            # remove_stat = cv2.resize(remove_stat, (0,0), fx=0.5, fy=0.5)
            remove_stat = cv2.resize(remove_stat, (self.input_size, self.input_size))
            img = remove_stat / np.max(remove_stat)
            # img = Brightness_Correction(img)
            img = np.power(img, (1.0/2.2))
            img = img.transpose(2,0,1)
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(gt.copy()).float()
        si = torch.from_numpy(si.copy()).float()

        return img, gt, si #, camera
    
    def generate_statistic_gt(self, img, ill):
        if not self.statistic_mode:
            return img, ill, np.ones_like(ill)
        remove_stat, stat = self.mapping_into_statis(img)
        # mapping ill to statisic
        true_stat = self.L2_norm(ill * stat)
        return remove_stat, true_stat, stat
    
    def load_jpg(self, img_path):
        img = np.load(img_path).astype(np.float32)[:,:,::-1]
        img = img / np.max(img)
        R = remove_gamma(img)
        if self.mode == 'train' and self.camera_trans != None:
            if self.camera_trans == 'all':
                camera = self.camera_mode[np.random.randint(0, 11)] 
            else:
                camera = self.camera_trans
        else:
            camera = 'JPG'      
        R = sRGB2Camera(R, camera)
        return R, camera
    
    def load_raw(self, img_path):
        img = np.load(img_path + '.npy').astype(np.float32)[:,:,::-1]
        mask = np.load(img_path + '_mask.npy').astype(np.bool)
        ill = np.load(img_path + '_gt.npy').astype(np.float32)
        camera = str(np.load(img_path + '_camera.npy'))
        # preprocess raw
        img = img / np.max(img)
        idx1, idx2, _ = np.where(mask == False)
        img[idx1, idx2, :] = 0 # 1e-5
        return img, ill, camera
    
    def load_cube_jpg(self, img_path):
        img = np.load(img_path + '.npy').astype(np.float32)[:,:,::-1]
        mask = np.load(img_path + '_mask.npy').astype(np.bool)
        img = img / np.max(img)
        img = remove_gamma(img)
        idx1, idx2, _ = np.where(mask == False)
        img[idx1, idx2, :] = 0 # 1e-5
        if self.mode == 'train' and self.camera_trans != None:
            if self.camera_trans == 'all':
                camera = self.camera_mode[np.random.randint(0, 11)] 
            else:
                camera = self.camera_trans
        else:
            camera = 'JPG'      
        img = sRGB2Camera(img, camera)
        return img, camera
        
    def three_fold(self, idx):
        img_list = []

        if self.mode == 'train':
            for ds, mt in zip(self.dataset_name, self.multiple):
                # Mix JPG
                if ds != 'JPG':
                    for i in range(3):
                        if i == idx:
                            continue
                        temp = self.load_nameseq(self.data_dir + '/{}_fold{}.txt'.format(ds, i))
                        if ds == 'Cube_half':
                            temp = self.mix_raw_jpg(temp, get_jpg=False)
                        elif ds == 'Cube_jpg':
                            temp = self.mix_raw_jpg(temp, get_jpg=True)
                        img_list += mt * temp
                else:
                    jpg_lst = self.load_nameseq(self.data_dir + '/NPlace205_train.txt')
                    random.shuffle(jpg_lst)
                    jpg_lst = jpg_lst[:6000]
                    img_list += jpg_lst
            random.shuffle(img_list)
        else:
            assert len(self.dataset_name) == 1
            if self.dataset_name[0] in ['NUS_half', 'CC_half', 'Cube_half', 'NUS_ori', 'CC_ori', 'Cube_ori', 'demo']:
                img_list = self.load_nameseq(self.data_dir + '/{}_fold{}.txt'.format(self.dataset_name[0], idx))
            else:
                img_list = self.load_nameseq(self.data_dir + '/NPlace205_valid.txt')

        return img_list
    
    def mix_raw_jpg(self, img_list, get_jpg, split2jpg_ratio=0):
        ret_list = []
        
        img_array = np.array(img_list)
        random_state = np.random.RandomState(seed=1)
        indexes = np.arange(len(img_list))
        random_state.shuffle(indexes)
        
        if get_jpg:
            jpg_idx = indexes[:int(len(img_list) * split2jpg_ratio)]
            ret_list += list(map(lambda x:x.replace('half','jpg'), list(img_array[jpg_idx])))
        else:
            raw_idx = indexes[int(len(img_list) * split2jpg_ratio):]
            ret_list += list(img_array[raw_idx])
        # print(ret_list)
        return ret_list
        
    def augment_ill(self, img, illumination):
        ill_aug = (np.random.random(3) - 0.5) * self.AUG_COLOR + 1
        new_illum = illumination * ill_aug
        new_image = img * ill_aug
        new_illum = np.clip(new_illum, 0.0001, 1000)
        new_image = np.clip(new_image, 0, 65535)
        return new_image, new_illum
        
    def augment_img(self, ldr):
        angle = (random.random() - 0.5) * self.ANGLE
        scale = math.exp(random.random() * math.log(self.SCALE[1] / self.SCALE[0])) * self.SCALE[0]
        s = int(round(min(ldr.shape[:2]) * scale))
        s = min(max(s, 10), min(ldr.shape[:2]))
        start_x = random.randrange(0, ldr.shape[0] - s + 1)
        start_y = random.randrange(0, ldr.shape[1] - s + 1)        
        flip_lr = random.randint(0, 1) # Left-right flip?   
        def crop(img):
            if img is None:
                return None
            img = img[start_x:start_x + s, start_y:start_y + s]
            img = rotate_and_crop(img, angle)
            img = cv2.resize(img, (self.input_size, self.input_size))
            if flip_lr:
                img = img[:, ::-1]
            img = img.astype(np.float32)    
            return img           
        return crop(ldr)
