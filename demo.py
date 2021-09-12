#!/usr/bin/env python
# coding: utf-8
import torch
from torch.utils.data import DataLoader
import logging
import time
import argparse
import nni
from nni.utils import merge_parameter

from tools import AverageMeter, Dispatcher, reset_meters, error_evaluation
from datasets import MIX
from model import CGA, Angular_loss
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
    
def load_from_ckpt(model, path):
    pretrained = torch.load(path)
    model.load_state_dict(pretrained['net'][0])
    epoch = pretrained['epoch']
    step = pretrained['step']
    return model, epoch, step

def report_single_image(img):
    return

def test(model, criterion, loader, device):
    loss_lst = []
    # single test
    for img_val, gt_val, si_val in loader:
        img_val = img_val.to(device).float()
        gt_val = gt_val.to(device).float()
        si_val = si_val.to(device).float()
        pred_val = model(img_val)
        loss_val = criterion(pred_val / si_val, gt_val / si_val) 
        print(loss_val.item())
        loss_lst.append(loss_val)
    
    # evalution
    error_evaluation(loss_lst)

def main(args):
    exp_msg = "Start! Exp:{}".format(args.exp_name)
    logger.info(exp_msg)
    
    # preparing DATALOADER
    demo_loader = gen_loader(['Demo'], 'all', 'valid', args=args, multiple=[1])
    
    # preparing MODEL
    device = args.device
    model = CGA(normalization='CGIN').to(device)

    # preparing CRITERION
    criterion = Angular_loss()

    model, epoch_start, step = load_from_ckpt(model, args.load_ckpt)
    msg = "load pretrained model: {}, from epoch: {}, step: {}".format(args.load_ckpt, epoch_start, step)            
    logger.info(msg)
    
    disp = Dispatcher(step, epoch_start, args.exp_name, args.save_path)

    with torch.no_grad():
        model.eval()
        test(model, criterion, demo_loader, device)
    
    end_msg = 'Finish!'
    logger.info(end_msg)

if __name__ == '__main__':
    params = get_params()
    print(params)
    main(params)
