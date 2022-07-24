#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 20, 12
import cv2
import logging
import time
import argparse

from tools import AverageMeter, Dispatcher, reset_meters, error_evaluation
from datasets import MIX
from model import TLCC, Angular_loss
from thop import profile, clever_format

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='log/TLCC_test.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='HyperParam List')
    parser.add_argument('--data_path', default='./data/processed/')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input_size', default=512)
    parser.add_argument('--load_ckpt_fold0', default="None")
    parser.add_argument('--load_ckpt_fold1', default="None")
    parser.add_argument('--load_ckpt_fold2', default="None")
    parser.add_argument('--bright_occ_mode', default=False) # abandoned
    parser.add_argument('--blur_mode', default=False)  # abandoned
    parser.add_argument('--num_workers', default=8)
    parser.add_argument('--statistic_mode', type=bool, default=True)
    args, _ = parser.parse_known_args()
    return args

def gen_loader(dataset, fold_idx, args):
    _data = MIX(dataset, args.data_path, 'vaild', 
                camera_trans=None, 
                fold_idx=fold_idx,  
                input_size=args.input_size,
                statistic_mode = args.statistic_mode,
                bright_occ_mode=args.bright_occ_mode, 
                blur_mode=args.blur_mode
            )
    loader = DataLoader(
                    dataset=_data, 
                    batch_size=1, 
                    shuffle=True, 
                    num_workers=2
                )
    print(f'{dataset}:{len(_data)}')
    return loader

def get_loader(args):
    return {
        'CC_fold0': gen_loader(['CC_ori'], 0, args=args),
        'CC_fold1': gen_loader(['CC_ori'], 1, args=args),
        'CC_fold2': gen_loader(['CC_ori'], 2, args=args),
    }
        

def print_msg(Meter_dict, disp, mode):
    msg = 'E:{},S:{},M:{}'.format(disp.epoch+1, disp.step, mode)
    for name, m in Meter_dict.items():
        if name == 'Valid':
            continue
        if m.avg <= 0:
            continue
        msg += ', {}:{:.4f}'.format(name, m.avg)
    for name, val in disp.best_dict.items():
        msg += ', {}_v:{:.2f}'.format(name, val)
    msg += ', t:{:.2f}'.format(disp.time_cost())
    return msg
        

def load_from_ckpt(model, path):
    pretrained = torch.load(path)
    model.load_state_dict(pretrained['net'][0], strict=False)
    return model
    
def test(model, criterion, loader, device):
    loss_lst = []
    for img_val, gt_val, si_val in loader:
        img_val = img_val.to(device).float()
        gt_val = gt_val.to(device).float()
        si_val = si_val.to(device).float()

        pred_val = model(img_val)

        loss_val = criterion(pred_val / si_val, gt_val / si_val) 
        loss_lst.append(loss_val)

    return loss_lst
    
def print_model_flops(model, args):
    img = torch.randn(1, 3, args.input_size, args.input_size).float().to(args.device)
    flops, params = profile(model, inputs=(img, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('input_size: {}, flops: {}, params: {}'.format(args.input_size, flops, params))

def main(args):
    # preparing DATALOADER
    loader_dict = get_loader(args)
    
    # preparing MODEL
    device = args.device
    model = TLCC(normalization='CGIN').to(device)
    print_model_flops(model, args)

    # preparing CRITERION
    criterion = Angular_loss()

    st = time.time()
    d_val = []
    name_lst = ['CC_fold0', 'CC_fold1', 'CC_fold2']
    ckpt_path_lst = [args.load_ckpt_fold0, args.load_ckpt_fold1, args.load_ckpt_fold2]
    for name, ckpt_path in zip(name_lst, ckpt_path_lst):
        model = load_from_ckpt(model, ckpt_path)
        with torch.no_grad():
            test_loader = loader_dict[name]
            model.eval()
            angular_list = test(model, criterion, test_loader, device)
            d_val += angular_list
    error_evaluation(d_val)
    print("total cost:", time.time() - st)
    end_msg = 'Finish!'
    logger.info(end_msg)

if __name__ == '__main__':
    try:
        params = get_params()
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
