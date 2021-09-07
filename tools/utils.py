"""
Some tools, too messy ...
Later to arrange ...
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import math

def error_evaluation(error_list):
    es = error_list.copy()
    es = torch.stack(es)
    es = es.view((-1))
    es = es.to('cpu').numpy()
    es.sort()
    ae = np.array(es).astype(np.float32)

    x, y, z = np.percentile(ae, [25, 50, 75])
    Mean = np.mean(ae)
    Med = np.median(ae)
    Tri = (x+ 2 * y + z)/4
    T25 = np.mean(ae[:int(0.25 * len(ae))])
    L25 = np.mean(ae[int(0.75 * len(ae)):])

    print("Mean\tMedian\tTri\tBest 25%\tWorst 25%")  
    print("{:3f}\t{:3f}\t{:3f}\t{:3f}\t{:3f}".format(Mean, Med, Tri, T25, L25))
    
def show_tensor(tensor):
    tensor_show = tensor.to('cpu').numpy().transpose((0, 2, 3, 1))
    # tensor_show = apply_gamma(tensor_show)
    # tensor_show = np.clip(tensor_show, 0, 1)
    bn = tensor_show.shape[0]
    for i in range(bn):
        plt.imshow(tensor_show[i, : ,: ,:])
        plt.show()

def show_tensor_biased(tensor, gt):
    tensor = torch.pow(tensor, 2.2)
    bn = tensor.shape[0]
    tensor = tensor / gt.view((bn, 3, 1, 1)) / 1.73
    tensor_show = tensor.to('cpu').numpy().transpose((0, 2, 3, 1))
    tensor_show = np.power(tensor_show, 1/2.2)
    bn = tensor_show.shape[0]
    for i in range(bn):
        plt.imshow(tensor_show[i, : ,: ,:])
        plt.show()

def apply_gamma(img):
    T = 0.0031308
    # rgb1 = torch.max(rgb, rgb.new_tensor(T))
    return np.where(img < T, 12.92 * img, (1.055 * np.power(np.abs(img), 1 / 2.4) - 0.055))

def remove_gamma(img):
    T = 0.04045
    #img1 = np.max(img, T)
    #print(img1.shape)
    return np.where(img < T, img / 12.92, np.power(np.abs(img + 0.055) / 1.055, 2.4))
    
def stat2uv(lst):
    lst = torch.stack(lst)
    lst = lst.view((-1, 3))
    u = lst[:, 0] / lst[:, 1]
    v = lst[:, 2] / lst[:, 1]
    return torch.stack([u, v])

def compute_ang_std(lst):
    lst = torch.stack(lst)
    point_lst = lst.view((-1, 3))
    rgb_vec = torch.nn.functional.normalize(point_lst, 2, 1)
    rgb_mean = torch.mean(rgb_vec, 0, keepdim=True)
    rgb_norm = torch.nn.functional.normalize(rgb_mean, 2, 1)
    arccos_num = torch.sum(rgb_vec * rgb_norm, 1)
    arccos_num = torch.clamp(arccos_num, -1, 1)
    angle = torch.acos(arccos_num) * (180 / math.pi)
    dev = torch.std(angle)
    return dev

def compute_uv_std(uv):
    uv_mean = torch.mean(uv, 0, keepdim=True)
    # print(uv_mean.shape, uv_mean)
    dist = torch.functional.norm(uv - uv_mean, 2, 1)
    # print(dist.shape, dist[:10])
    return torch.std(dist)

def get_grey_map_torch(img, angular):
    norm = F.normalize(img)
    prob = torch.sum(norm, 1, keepdim=True) / math.sqrt(3)
    bin_map = torch.where(prob >= np.cos(math.pi / 180 * angular), 1., 0.)
    return bin_map

def display_s(input_rgb, target, mask):
    bn, c = target.shape
    gt_rgb = input_rgb * target.view((bn, c, 1, 1))
    gt_rgb_show = gt_rgb.to('cpu').numpy().transpose((0,2,3,1))
    gt_rgb_show = apply_gamma(gt_rgb_show)
    gt_rgb_show /= np.max(gt_rgb_show, (1,2,3), keepdims=True)
    input_rgb_show = input_rgb.to('cpu').numpy().transpose((0,2,3,1))
    weight_map = get_grey_map_torch(gt_rgb * mask)
    weight_map = weight_map.numpy()
    for i in range(1):
        plt.subplot(121)
        plt.imshow(input_rgb_show[i, :, :, :])

        plt.subplot(122)
        plt.imshow(gt_rgb_show[i, :, :, :])
        plt.show()  

        plt.imshow(weight_map[i, 0, :, :])
        plt.show() 

def diff_iim(ori, bal):
    ori = ori.double()
    bal = bal.double()
    print(torch.max(torch.abs(ori)), torch.max(torch.abs(bal)), torch.max(torch.abs(bal - ori)))
    x = torch.where(torch.abs(bal - ori) > 1e-6)[0]
    print("diff pixel num: {}".format(x.shape[0]))
    zero_pixel = torch.where(bal == 0)[0].shape
    return zero_pixel

def Brightness_Correction(img):
    gray = np.sum(img * np.array([0.299, 0.587, 0.114,]), 2)
    gray_scale = 0.25 / (np.mean(gray) + 1e-15)
    bright = img * gray_scale
    return bright

def get_ccm(camera_model):
    # extracted from dcraw.c
    matrices = {'Canon5D':      (6347,-479,-972,-8297,15954,2480,-1968,2131,7649),  # Canon 5D
                'Canon1D':      (4374,3631,-1743,-7520,15212,2472,-2892,3632,8161), # Canon 1Ds
                'Canon550D':    (6941,-1164,-857,-3825,11597,2534,-416,1540,6039),  # Canon 550D
                'Canon1DsMkIII':(5859,-211,-930,-8255,16017,2353,-1732,1887,7448),  # Canon 1Ds Mark III
                'Canon600D':    (6461,-907,-882,-4300,12184,2378,-819,1944,5931),   # Canon 600D
                'FujifilmXM1':  (10413,-3996,-993,-3721,11640,2361,-733,1540,6011), # FujifilmXM1
                'NikonD5200':   (8322,-3112,-1047,-6367,14342,2179,-988,1638,6394), # Nikon D5200
                'OlympusEPL6':  (8380,-2630,-639,-2887,10725,2496,-627,1427,5438),  # Olympus E-PL6
                'PanasonicGX1': (6763,-1919,-863,-3868,11515,2684,-1216,2387,5879), # Panasonic GX1
                'SamsungNX2000':(7557,-2522,-739,-4679,12949,1894,-840,1777,5311),  # SamsungNX2000
                'SonyA57':      (5991,-1456,-455,-4764,12135,2980,-707,1425,6701)}  # Sony SLT-A57
    xyz2cam = np.asarray(matrices[camera_model]) / 10000
    xyz2cam = xyz2cam.reshape(3, 3)
    xyz2cam = xyz2cam / np.sum(xyz2cam, axis=1, keepdims=True)
    
    linsRGB2XYZ = np.array(((0.4124564, 0.3575761, 0.1804375), 
                            (0.2126729, 0.7151522, 0.0721750),
                            (0.0193339, 0.1191920, 0.9503041)))
    linsRGB2XYZ = linsRGB2XYZ / np.sum(linsRGB2XYZ, axis=1, keepdims=True)
    linsRGB2cam = xyz2cam.dot(linsRGB2XYZ)
    # cam2linsRGB = np.linalg.inv(linsRGB2cam_norm)
    return linsRGB2cam

def img_witch_ccm(img, camera_model):
    mat = get_ccm(camera_model)
    raw = mat[np.newaxis, np.newaxis, :, :] * img[:, :, np.newaxis, :]
    raw = np.sum(raw, axis=-1)
    raw = np.clip(raw, 0., 1.)
    return raw

def get_camera_mat(camera_model, ToCamera=True):
    # extracted from dcraw.c
    matrices = {'Canon5D':      (6347,-479,-972,-8297,15954,2480,-1968,2131,7649),  # Canon 5D
                'Canon1D':      (4374,3631,-1743,-7520,15212,2472,-2892,3632,8161), # Canon 1Ds
                'Canon550D':    (6941,-1164,-857,-3825,11597,2534,-416,1540,6039),  # Canon 550D
                'Canon1DsMkIII':(5859,-211,-930,-8255,16017,2353,-1732,1887,7448),  # Canon 1Ds Mark III
                'Canon600D':    (6461,-907,-882,-4300,12184,2378,-819,1944,5931),   # Canon 600D
                'FujifilmXM1':  (10413,-3996,-993,-3721,11640,2361,-733,1540,6011), # FujifilmXM1
                'NikonD5200':   (8322,-3112,-1047,-6367,14342,2179,-988,1638,6394), # Nikon D5200
                'OlympusEPL6':  (8380,-2630,-639,-2887,10725,2496,-627,1427,5438),  # Olympus E-PL6
                'PanasonicGX1': (6763,-1919,-863,-3868,11515,2684,-1216,2387,5879), # Panasonic GX1
                'SamsungNX2000':(7557,-2522,-739,-4679,12949,1894,-840,1777,5311),  # SamsungNX2000
                'SonyA57':      (5991,-1456,-455,-4764,12135,2980,-707,1425,6701)}  # Sony SLT-A57
    xyz2cam = np.asarray(matrices[camera_model]) / 10000
    xyz2cam = xyz2cam.reshape(3, 3)
    xyz2cam = xyz2cam / np.sum(xyz2cam, axis=1, keepdims=True)
    if ToCamera:
        return xyz2cam
    else:
        return np.linalg.inv(xyz2cam)

def get_lin2XYZ_cam(ToXYZ=True):
    linsRGB2XYZ = np.array(((0.4124564, 0.3575761, 0.1804375),
                            (0.2126729, 0.7151522, 0.0721750),
                            (0.0193339, 0.1191920, 0.9503041)))
    linsRGB2XYZ = linsRGB2XYZ / np.sum(linsRGB2XYZ, axis=1, keepdims=True)
    if ToXYZ:
        return linsRGB2XYZ
    else:
        return np.linalg.inv(linsRGB2XYZ)

def sRGB2Camera(img, Camera):
    if Camera in ['JPG', None]:
        return img
    lin2XYZ = get_lin2XYZ_cam(True)
    XYZ2camera = get_camera_mat(Camera, True)
    img_xyz = img.dot(lin2XYZ.T)
    img_cam = img_xyz.dot(XYZ2camera.T)
    img_cam = np.clip(img_cam, 0., 1.)
    return img_cam

def print_msg(Meters, dataset_name, epoch, step, mode, best_lst,time_use):
    msg = 'E:{},S:{},M:{}'.format(epoch+1, step, mode)
    for m, name in zip(Meters, dataset_name):
        msg += ', {}:{:.4f}'.format(name, m.avg)
    for val, name in zip(best_lst, ['CC','NUS','Place']):
        msg += ', {}_v:{:.2f}'.format(name, val)
    msg += ', t:{:.2f}'.format(time_use)
    return msg

def reset_meters(meter_lst):
    for _, m in meter_lst.items():
        m.reset()

def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
                # params_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    print(len(list(module.parameters())), len(params_decay), len(params_no_decay))
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay

def print_parameters_info(parameters):
    for k, param in enumerate(parameters):
        print('[{}/{}] {}'.format(k+1, len(parameters), param.shape))
