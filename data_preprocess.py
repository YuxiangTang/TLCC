import os
import time

import numpy as np
import cv2
from scipy.io import loadmat
import argparse
"""
TODO: add colorchart detection to get ccm.
"""
# detector = cv2.mcc.CCheckerDetector_create()
# def detect_colorchart(img, gt, detector, img_path):
#     uint_img = (np.power(img[:, :, ::-1] / gt / np.max(img), 1/2.4) * 255).astype(np.uint8)
#     if detector.process(uint_img, cv2.mcc.MCC24, nc=1):
#         checker = detector.getBestColorChecker()
#         chartsRGB = checker.getChartsRGB()
#         mat24 = chartsRGB[:,1].copy().reshape(24, 3)
#         mat24 /= np.max(mat24)
#     else:
#         raise Exception("Detection is failed, img path:", img_path)
#     # mat24 is RGB format
#     return mat24


def preprocess_colorchecker(data_dir, output_path, resize2half):
    """
    Preprocess the data from the ColorChecker dataset.
    black/white level correction -> generate mask, camera type
    
    :param data_dir: source data path
    :param output_path: output data path
    
    save npy: img, mask, camera type, ground truth
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_gt = load_ill_ccd(data_dir)
    img_list = load_nameseq(data_dir + 'img.txt')
    lst = []
    SATURATION_SCALE = 0.95

    start = time.time()
    for idx in range(0, len(img_list)):
        #  read image
        img_path = data_dir + img_list[idx] + ".png"

        img = cv2.imread(img_path, -1).astype(np.float32)
        h, w, c = img.shape

        # read coordinate & generate mask
        coor = load_mcc_ccd(data_dir, img_list[idx], w, h)
        mask = np.ones((h, w))
        mask = cv2.fillPoly(mask, [coor], (0, 0, 0)).reshape((h, w, 1))

        # black level correction
        if img_list[idx].startswith('IMG'):
            blackLevel = 129
            camera = 'Canon5D'
        else:
            blackLevel = 1
            camera = 'Canon1D'

        img = img - blackLevel
        img[img < 0] = 0
        # saturationLevel= (3692 - blackLevel) * SATURATION_SCALE
        # img[img > saturationLevel] = saturationLevel

        # detect colorchecker
        # TODO: get color chart
        # mat24 = detect_colorchart(img, img_gt[idx], detector, img_path)

        if resize2half:
            # resize image, (* 4) to be integer.
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) * 4
            mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5) * 4

        # In order to reduce file size.
        img = img.astype(np.uint16)
        h, w, c = img.shape
        mask = mask.astype(np.bool_).reshape((h, w, 1))

        # write file name
        print("[CCD Running] idx:{}, path:{}, camera:{}, ill:{}".format(
            idx, img_path, camera, img_gt[idx]))
        np.save('{}/{}.npy'.format(output_path, img_list[idx]), img)
        np.save('{}/{}_mask.npy'.format(output_path, img_list[idx]), mask)
        np.save('{}/{}_camera.npy'.format(output_path, img_list[idx]), camera)
        np.save('{}/{}_gt.npy'.format(output_path, img_list[idx]), img_gt[idx])
        # np.save('{}/{}_mat24.npy'.format(output_path, img_list[idx]), mat24)
    print("CCD data is finished! time cost: {:2f}s".format(time.time() - start))


def preprocess_nus(data_dir, output_path, resize2half):
    """
    Preprocess the data from the NUS-8 dataset.
    black/white level correction -> generate mask, camera type
    
    :param data_dir: source data path
    :param output_path: output data path
    
    save npy: img, mask, camera type, ground truth
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = load_data_nus(data_dir)
    dynamic = {
        'Canon1DsMkIII': 16383,
        'Canon600D': 16383,
        'FujifilmXM1': 4095,
        'NikonD5200': 16383,
        'OlympusEPL6': 4095,
        'PanasonicGX1': 4095,
        'SamsungNX2000': 4095,
        'SonyA57': 4095
    }

    SATURATION_SCALE = 0.95

    start = time.time()
    for idx in range(0, len(data)):
        # read meta data
        img_path = data[idx]['imgpath']
        saturationLevel = data[idx]['saturation_level']
        darkness_level = data[idx]['darkness_level']
        gt = data[idx]['gt']
        mcc = data[idx]['mcc']
        camera = data[idx]['camera']
        dynum = dynamic[camera]

        img = cv2.imread(img_path, -1).astype(np.float32)

        # black / white level correction
        img = img - darkness_level
        img[img < 0] = 0
        h, w, c = img.shape
        sat = (saturationLevel - darkness_level) * SATURATION_SCALE
        img[img > sat] = sat
        img = np.clip(img, 0, 65535)

        # generate mask
        coor = [[mcc[2], mcc[0]], [mcc[2], mcc[1]], [mcc[3], mcc[1]], [mcc[3], mcc[0]]]
        coor = np.array(coor).astype(np.int32)
        mask = np.ones((h, w)).astype(np.float64)
        mask = cv2.fillPoly(mask, [coor], (0, 0, 0))

        # detect colorchecker
        # mat24 = detect_colorchart(img, gt, detector, img_path)

        if resize2half:
            # resize image, (* 4) to be integer.
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) * 4
            mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5) * 4
        h, w, c = img.shape
        mask = mask.astype(np.bool_).reshape((h, w, 1))
        img = img.astype(np.uint16)

        # save image
        print("[NUS8 Running] idx:{}, path:{}, camera:{}, ill:{}".format(idx, img_path, camera, gt))

        np.save('{}/{}.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), img)
        np.save('{}/{}_mask.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), mask)
        np.save('{}/{}_camera.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), camera)
        np.save('{}/{}_gt.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), gt)
        # np.save('{}/{}_mat24.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), mat24)
    print("NUS-8 data is finished! time cost: {:2f}s".format(time.time() - start))


def preprocess_cube(data_dir, output_path, resize2half):
    """
    Preprocess the data from the Cube+ dataset.
    black/white level correction -> generate mask, camera type
    
    :param data_dir: source data path
    :param output_path: output data path
    
    save npy: img, mask, camera type, ground truth
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_gt = load_ill_cube(data_dir)
    img_list = load_nameseq(data_dir + 'img.txt')

    start = time.time()
    for idx in range(0, len(img_list)):
        img_path = data_dir + img_list[idx]
        img = cv2.imread(img_path, -1).astype(np.float32)

        gt = img_gt[idx]
        camera = 'Canon550D'
        saturationLevel = np.max(img) - 2
        blackLevel = 2048
        img = img - blackLevel
        img[img < 0] = 0
        img[img > saturationLevel - blackLevel] = saturationLevel - blackLevel

        # generate mask
        mask = np.ones_like(img)[:, :, 0]
        mask[1050:, 2050:] = 0

        if resize2half:
            # resize image, (* 4) to be integer.
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) * 4
            mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5) * 4

        h, w, c = img.shape
        # save image
        mask = mask.astype(np.bool_).reshape((h, w, 1))
        img = img.astype(np.uint16)

        print("[Cube+ Running] idx:{}, path:{}, camera:{}, ill:{}".format(
            idx, img_path, camera, gt))

        np.save('{}/{}.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), img)
        np.save('{}/{}_mask.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), mask)
        np.save('{}/{}_camera.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), camera)
        np.save('{}/{}_gt.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), gt)
    print("Cube+ data is finished! time cost: {:2f}s".format(time.time() - start))


def load_ill_cube(data_dir):
    img_gt = []
    with open(data_dir + "cube+_gt.txt", "r") as f:
        for line in f:
            line = line.rstrip()
            item = list(map(float, line.split(' ')))
            img_gt.append(item)
    img_gt = np.array(img_gt)
    img_gt = img_gt / np.linalg.norm(img_gt, ord=2, axis=1).reshape(-1, 1)
    return img_gt


def load_ill_ccd(data_dir):
    img_gt = loadmat(data_dir + '/real_illum_568..mat')['real_rgb']
    img_gt = img_gt / np.linalg.norm(img_gt, ord=2, axis=1).reshape(-1, 1)
    return img_gt


def load_mcc_ccd(data_dir, fn, w, h):
    with open(data_dir + '/coordinates/{}_macbeth.txt'.format(fn)) as f:
        lines = f.readlines()
        width, height = map(float, lines[0].split())
        scale_x = 1 / width
        scale_y = 1 / height
        lines = [list(map(float, lines[i].strip().split(' '))) for i in [1, 2, 4, 3]]
        polygon = np.array(lines).astype(np.float32)
        polygon *= np.array([w * scale_x, h * scale_y], dtype=np.float32)
    return np.array(polygon, dtype=np.int32)


def load_data_nus(data_dir):
    img_list = []
    k = 0
    camera_list = [
        "Canon1DsMkIII", "Canon600D", "FujifilmXM1", "NikonD5200", "OlympusEPL6", "PanasonicGX1",
        "SamsungNX2000", "SonyA57"
    ]
    for camera in camera_list:
        mat = loadmat(data_dir + camera + '/' + camera + '_gt.mat')
        for i in range(mat['all_image_names'].shape[0]):
            img_p = data_dir + camera + '/' + mat['all_image_names'][i][0][0] + ".PNG"
            #if camera not in img_list:
            #img_list[camera] = []
            img_list.append({
                'imgpath': img_p,
                'saturation_level': mat['saturation_level'][0][0],
                'camera': camera,
                'darkness_level': mat['darkness_level'][0][0],
                'gt': mat['groundtruth_illuminants'][i],
                'mcc': mat['CC_coords'][i],
                'k': k
            })
            k += 1
    return img_list


def load_nameseq(dir_path):
    img_list = []
    with open(dir_path, "r") as f:
        for line in f:
            line = line.rstrip()
            img_list.append(line)
    return img_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HyperParam List')
    parser.add_argument('--input_dir', type=str, default="./data/source/")
    parser.add_argument('--output_dir', type=str, default="./data/processed/")
    parser.add_argument('--resize2half', type=bool, default=False)
    args, _ = parser.parse_known_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    flag_ = args.resize2half
    print(args)

    data_dir = input_dir + '/colorchecker2010/'
    output_path = output_dir + '/CC/'
    preprocess_colorchecker(data_dir, output_path, flag_)

    data_dir = input_dir + '/NUS/'
    output_path = output_dir + '/NUS/'
    preprocess_nus(data_dir, output_path, flag_)

    data_dir = input_dir + '/Cube/'
    output_path = output_dir + '/Cube/'
    preprocess_cube(data_dir, output_path, flag_)