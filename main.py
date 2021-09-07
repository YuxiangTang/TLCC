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

from tools import AverageMeter, Dispatcher, reset_meters
from datasets import MIX
from model import CGAA, Angular_loss
from thop import profile,clever_format


logger = logging.getLogger()
fhandler = logging.FileHandler(filename='log/TLCC.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='HyperParam List')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',help='input batch size for training (default: 4)')
    parser.add_argument('--aug_num', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
    parser.add_argument('--num_epochs', type=int, default=150, metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--data_path', default='/dataset/colorconstancy/quick_data/')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_path', default='./ckpt/')
    parser.add_argument('--input_size', default=512)
    parser.add_argument('--exp_name', default='TLCC_try4')
    parser.add_argument('--fold_idx', default=0)
    parser.add_argument('--load_ckpt', default="None")
    parser.add_argument('--bright_occ_mode', default=False)
    parser.add_argument('--blur_mode', default=False)
    parser.add_argument('--num_workers', default=8)
    parser.add_argument('--step', default=0)
    args, _ = parser.parse_known_args()
    return args

def get_loader(args):
    RAW_train_data = MIX('NJPG', args.data_path, 'train', camera_trans=None, fold_idx=args.fold_idx, aug_num=args.aug_num,
                         input_size=args.input_size, bright_occ_mode=args.bright_occ_mode, blur_mode=args.blur_mode)
    RAW_train_loader = DataLoader(dataset=RAW_train_data, batch_size=args.batch_size, shuffle=True, 
                         num_workers=args.num_workers, drop_last=True, prefetch_factor=20, persistent_workers=True, pin_memory=True)

    JPG_train_data = MIX('JPG', args.data_path, 'train', camera_trans='all', fold_idx=args.fold_idx, aug_num=args.aug_num,
                         input_size=args.input_size, bright_occ_mode=args.bright_occ_mode, blur_mode=args.blur_mode)
    JPG_train_loader = DataLoader(dataset=JPG_train_data, batch_size=args.batch_size, shuffle=True, 
                         num_workers=args.num_workers, drop_last=True, prefetch_factor=20, persistent_workers=True, pin_memory=True)

    JPG_vaild_data = MIX('JPG', args.data_path, 'vaild', camera_trans=None, fold_idx=args.fold_idx,  
                         input_size=args.input_size, bright_occ_mode=args.bright_occ_mode, blur_mode=args.blur_mode)
    JPG_vaild_loader = DataLoader(dataset=JPG_vaild_data, batch_size=1, shuffle=True, num_workers=2)

    CC_vaild_data = MIX('CC', args.data_path, 'vaild', fold_idx=args.fold_idx,  
                        input_size=args.input_size, bright_occ_mode=args.bright_occ_mode, blur_mode=args.blur_mode)
    CC_vaild_loader = DataLoader(dataset=CC_vaild_data, batch_size=1, shuffle=True, num_workers=2)

    NUS_vaild_data = MIX('NUS', args.data_path, 'vaild', fold_idx=args.fold_idx,  
                         input_size=args.input_size, bright_occ_mode=args.bright_occ_mode, blur_mode=args.blur_mode)
    NUS_vaild_loader = DataLoader(dataset=NUS_vaild_data, batch_size=1, shuffle=True, num_workers=2)

    return RAW_train_loader, JPG_train_loader, JPG_vaild_loader, CC_vaild_loader, NUS_vaild_loader

def print_msg(Meter_dict, disp, mode):
    msg = 'E:{},S:{},M:{}'.format(disp.epoch+1, disp.step, mode)
    for name, m in Meter_dict.items():
        if name == 'Valid':
            continue
        msg += ', {}:{:.4f}'.format(name, m.avg)
    for name, val in disp.best_dict.items():
        msg += ', {}_v:{:.2f}'.format(name, val)
    msg += ', t:{:.2f}'.format(disp.time_cost())
    
    return msg
        
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_from_pretrained(model):
    name_map = dict()
    with open('map.txt') as f:
        for line in f:
            line = line.rstrip()
            item = line.split('\t')
            name_map[item[0]] = item[1]

    model_dict = model.squeezenet.state_dict() 
    pretrained_dict = model_dict

    pretrained = torch.load('./pretrained/squeezenet1_1.pth')
    for k,v in pretrained.items():
        if k in name_map:
            pretrained_dict[name_map[k]] = v

    model_dict.update(pretrained_dict)
    model.squeezenet.load_state_dict(model_dict)
    return model
    
def load_from_ckpt(model, optimizer, path):
    pretrained = torch.load(path)
    model.load_state_dict(pretrained['net'][0])
    optimizer.load_state_dict(pretrained['optimizer'])
    epoch = pretrained['epoch']
    step = pretrained['step']
    return model, optimizer, epoch, step

def save_ckpt(model, optimizer, disp, ds_name):
    state = {'net': [model.state_dict()],
            'optimizer' : optimizer.state_dict(), 
            'epoch' : disp.epoch,
            'step' : disp.step}
    torch.save(state, ckpt_path + '{}_{}_best.pth'.format(disp.exp_name, ds_name))
    
def train(training_object, model, optimizer, criterion, loader, Meter_dict, disp, device):
    for img, gt in loader:
        optimizer.zero_grad()

        img = img.to(device, non_blocking=True).float()
        gt  = gt.to(device, non_blocking=True).float()
        _, _, c, h, w = img.shape
        img, gt = img.view((-1, c, h, w)), gt.view((-1, 3))

        pred = model(img)
        ang_loss = criterion(pred, gt)

        if training_object == 'RAW_AL':
            loss = ang_loss
        else:
            loss = ang_loss
            
        loss.backward()
        Meter_dict[training_object].update(ang_loss.item())
        optimizer.step()

        disp.step += 1
        if disp.step  % 250 == 0:
            msg = print_msg(Meter_dict, disp, 'Train')
            logger.info(msg)
            print (msg)
            reset_meters(Meter_dict)
            
def vaild(ds_name, model, optimizer, criterion, loader, Meter_dict, disp, device):
    Meter_dict['Valid'].reset()
    loss_lst = []
    for img_val, gt_val in loader:
        img_val = img_val.to(device).float()
        gt_val = gt_val.to(device).float()

        pred_val = model(img_val)
        loss_val = criterion(pred_val, gt_val) 
        vaild_loss.update(loss_val.item())
        loss_lst.append(loss_val)

    # save best ckpt
    if disp.best_dict[ds_name] > vaild_loss.avg:
        disp.best_dict[ds_name] = vaild_loss.avg
        save_ckpt(model, optimizer, disp, ds_name)

    msg = print_msg(Meter_dict, disp, 'Valid')
    logger.info(msg)
    print(msg)
    error_evaluation(loss_lst)
    
def print_model_flops(model, device):
    img = torch.randn(1, 3, 512, 512).float().to(device)
    flops, params = profile(model, inputs=(img, camera_id))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops: {}, params: {}'.format(flops, params))

def main(args):
    exp_msg = "Start! Exp:{}".format(args.exp_name)
    print(exp_msg)
    logger.info(exp_msg)
    
    # preparing DATALOADER
    RAW_train_loader, JPG_train_loader, JPG_valid_loader, CC_valid_loader, NUS_valid_loader = get_loader(args)
    device = args.device
   
    # preparing MODEL
    model = CGAA().to(device)
    print(model)
    print_model_flops(model, device)
    exit()
    # preparing OPTIMIZER
    optimizer = torch.optim.Adam([{'params':model.parameters() , 'lr':args.lr}])

    # preparing CRITERION
    criterion = Angular_loss()

    epoch_start, step = 0, 0
    if args.load_ckpt != 'None': # Train from ckpt
        model, optimizer, epoch_start, step = load_from_ckpt(model, optimizer, args.load_ckpt)
        msg = "load pretrained model: {}, from epoch: {}, step: {}".format(args.load_ckpt, epoch_start, step)
        print(msg)
        logger.info(msg)
    else:  # Train from scratch
        model = load_from_pretrained(model)
        msg = "load model from scratch."
        print(msg)
        logger.info(msg)
    
    Meter_dict = {'JPG_AL':AverageMeter(),
                 'RAW_AL':AverageMeter(),
                 'Valid':AverageMeter()}

    reset_meters(Meter_dict)
    disp = Dispatcher(step, epoch_start, args.exp_name)
    
    for epoch in range(epoch_start, args.num_epochs): # max to num_epochs
        model.train()
        disp.time_start()
        disp.epoch = epoch
        # source domain learning
        if epoch < 30:
            train('JPG_AL', model, optimizer, criterion, JPG_train_loader, Meter_dict, disp, device)
        
        # target domain learning
        for _ in range(5):
            train('RAW_AL', model, optimizer, criterion, RAW_train_loader, Meter_dict, disp, device)
            break
            
        # Vaildation
        with torch.no_grad():
            model.eval()
            dataset_name = ['CC', 'NUS']
            loader_lst = [CC_valid_loader, NUS_valid_loader]
            for idx, ds_and_loader in enumerate(zip(dataset_name, loader_lst)):
                ds, loader = ds_and_loader
                vaild(ds, model, optimizer, criterion, loader, Meter_dict, disp, device)

        # Timing ckpt
        if epoch % 10 == 0:
            save_ckpt(model, optimizer, disp, 'uni')
    
    end_msg = 'Finish!'
    print(end_msg)
    logger.info(end_msg)

if __name__ == '__main__':
    try:
        # tuner_params = nni.get_next_parameter()
        # logger.debug(tuner_params)
        # params = vars(merge_parameter(get_params(), tuner_params))
        params = get_params()
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise