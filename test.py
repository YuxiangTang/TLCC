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
import nni
from nni.utils import merge_parameter

from tools import AverageMeter, Dispatcher, reset_meters, error_evaluation
from datasets import MIX
from model import TLCC, Angular_loss
from thop import profile, clever_format

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='log/TLCC.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='HyperParam List')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',help='input batch size for training (default: 4)')
    parser.add_argument('--aug_num', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.003)')
    parser.add_argument('--num_epochs', type=int, default=300, metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--data_path', default='/dataset/colorconstancy/quick_data/')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_path', default='./ckpt/')
    parser.add_argument('--input_size', default=512)
    parser.add_argument('--exp_name', default='test')
    parser.add_argument('--fold_idx', default=2)
    parser.add_argument('--load_ckpt', default="None")
    parser.add_argument('--bright_occ_mode', default=False)
    parser.add_argument('--blur_mode', default=False)
    parser.add_argument('--num_workers', default=12)
    parser.add_argument('--step', default=0)
    parser.add_argument('--finetuning', type=bool, default=False)
    args, _ = parser.parse_known_args()
    return args

def gen_loader(dataset, camera_trans, mode, args, multiple=[1]):
    assert mode in ['train', 'valid']
    if mode == 'train':
        _data = MIX(dataset, args.data_path, 'train', 
                        camera_trans=camera_trans,
                        fold_idx=args.fold_idx, 
                        aug_num=args.aug_num,
                        multiple=multiple,
                        input_size=args.input_size, 
                        bright_occ_mode=args.bright_occ_mode, 
                        blur_mode=args.blur_mode
                        )
        loader = DataLoader(
                        dataset=_data, 
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        num_workers=args.num_workers, 
                        drop_last=True, 
                        prefetch_factor=20, 
                        persistent_workers=True
                        )
    else:
        _data = MIX(dataset, args.data_path, 'vaild', 
                        camera_trans=None, 
                        fold_idx=args.fold_idx,  
                        input_size=args.input_size,
                        bright_occ_mode=args.bright_occ_mode, 
                        blur_mode=args.blur_mode)
        loader = DataLoader(
                        dataset=_data, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=2)
    print(f'{dataset}_{mode}:{len(_data)}')
    return loader

def get_loader(args, multiple=[5]):
    return {
        'JPG': (
            gen_loader(['JPG'], 'all', 'train', args=args, multiple=[1]),
            gen_loader(['JPG'], 'all', 'valid', args=args)
            ),
        'Cube': (
            gen_loader(['Cube_half'], None, 'train', args=args, multiple=multiple),
            gen_loader(['Cube_half'], None, 'valid', args=args)
            ),
        'NUS': (
            gen_loader(['NUS_half'], None, 'train', args=args, multiple=multiple),
            gen_loader(['NUS_half'], None, 'valid', args=args)
            ),
        'CC': (
            gen_loader(['CC_half'], None, 'train', args=args, multiple=multiple),
            gen_loader(['CC_half'], None, 'valid', args=args)
            ),
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

def save_ckpt(model, optimizer, disp, ds_name, mode):
    assert mode in ['schedule', 'best']
    state = {'net': [model.state_dict()],
            'optimizer' : optimizer.state_dict(), 
            'epoch' : disp.epoch,
            'step' : disp.step}
    if mode == 'best':
        torch.save(state, disp.ckpt_path + '{}_{}_best.pth'.format(disp.exp_name, ds_name))
    else:
        torch.save(state, disp.ckpt_path + '{}_{}.pth'.format(disp.exp_name, disp.epoch))
    
def train(training_object, model, optimizer, criterion, loader, Meter_dict, disp, device):
    for img, gt, _ in loader:
        optimizer.zero_grad()

        img = img.to(device).float()
        gt  = gt.to(device).float()
        _, _, c, h, w = img.shape
        img, gt = img.view((-1, c, h, w)), gt.view((-1, 3))

        pred = model(img)
        ang_loss = criterion(pred, gt)

        loss = ang_loss
            
        loss.backward()
        Meter_dict[training_object].update(ang_loss.item())
        optimizer.step()

        disp.step += 1
        if disp.step % 100 == 0:
            msg = print_msg(Meter_dict, disp, 'Train')
            logger.info(msg)
            reset_meters(Meter_dict)
            
def vaild(ds_name, model, optimizer, criterion, loader, Meter_dict, disp, device):
    Meter_dict['Valid'].reset()
    loss_lst = []
    for img_val, gt_val, si_val in loader:
        img_val = img_val.to(device).float()
        gt_val = gt_val.to(device).float()
        si_val = si_val.to(device).float()

        pred_val = model(img_val)

        loss_val = criterion(pred_val / si_val, gt_val / si_val) 
        Meter_dict['Valid'].update(loss_val.item())
        loss_lst.append(loss_val)

    # save best ckpt
    if disp.best_dict[ds_name] > Meter_dict['Valid'].avg:
        disp.best_dict[ds_name] = Meter_dict['Valid'].avg
        save_ckpt(model, optimizer, disp, ds_name, 'best')

    msg = print_msg(Meter_dict, disp, 'Valid')
    logger.info(msg)
    error_evaluation(loss_lst)
    
def print_model_flops(model, args):
    img = torch.randn(1, 3, args.input_size, args.input_size).float().to(args.device)
    flops, params = profile(model, inputs=(img, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('input_size: {}, flops: {}, params: {}'.format(args.input_size, flops, params))

def main(args):
    exp_msg = "Start! Exp:{}".format(args.exp_name)
    logger.info(exp_msg)
    
    # preparing DATALOADER
    loader_dict = get_loader(args)
    
    # preparing MODEL
    device = args.device
    model = TLCC(normalization='CGIN').to(device)
    from torchsummary import summary
    summary(model, (3, 512, 512))

    # print(model)
    print_model_flops(model, args)
    # preparing OPTIMIZER
    optimizer = torch.optim.Adam([{'params':model.parameters() , 'lr':args.lr}])

    # preparing CRITERION
    criterion = Angular_loss()

    epoch_start, step = 0, 0
    if args.load_ckpt != 'None': # Valid from ckpt
        model, optimizer, epoch_start, step = load_from_ckpt(model, optimizer, args.load_ckpt)
        msg = "load pretrained model: {}, from epoch: {}, step: {}".format(args.load_ckpt, epoch_start, step)            
    else:  # Train from scratch
        raise Exception('Please provide ckpt.')
    logger.info(msg)
    
    Meter_dict = {'JPG':AverageMeter(),
                  'Cube':AverageMeter(),
                  'NUS':AverageMeter(),
                  'CC':AverageMeter(),
                  'MIX':AverageMeter(),
                 'Valid':AverageMeter()}

    reset_meters(Meter_dict)
    disp = Dispatcher(step, epoch_start, args.exp_name, args.save_path)

    d_val = {}
    for name in ['CC']:
    # for name in ['NUS', 'CC', 'Cube']:
        with torch.no_grad():
            _, valid_loader = loader_dict[name]
            model.eval()
            st = time.time()
            vaild(name, model, optimizer, criterion, valid_loader, Meter_dict, disp, device)
            print("total cost:", time.time() - st)
            d_val[name] = Meter_dict['Valid'].avg
    nni.report_intermediate_result(d_val)
    
    end_msg = 'Finish!'
    logger.info(end_msg)

if __name__ == '__main__':
    try:
        tuner_params = nni.get_next_parameter()
        logger.info(tuner_params)
        params = merge_parameter(get_params(), tuner_params)
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise