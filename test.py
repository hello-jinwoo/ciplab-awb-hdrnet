import argparse
import os
import sys
from tqdm import tqdm
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from dataset import HDRDataset
from metrics import psnr
from model import HDRPointwiseNN
from utils import load_image, save_params, get_latest_ckpt, load_params
from error_measurement import mean_angular_error as mae
from visualize import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('--ckpt', type=str, default='ch/pretrained_model.pth', help='Model checkpoint path')
    parser.add_argument('--dataset', type=str, default='data_sample', help='Dataset path with input/output dirs')
    parser.add_argument('--net-input-size', type=int, default=128, help='Size of low-res input')
    parser.add_argument('--net-output-size', type=int, default=256, help='Size of full-res input/output')
    parser.add_argument('--bit-depth', type=int, default=10)

    params = vars(parser.parse_args())

    test_dataset = HDRDataset(params['dataset'], params=params, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    state_dict = torch.load(params['ckpt'])
    state_dict, model_params = load_params(state_dict)
    # model_params.update(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Run on CUDA")
    else:
        device = torch.device("cpu")
        print("Run on CPU")

    max_value = 2 ** params['bit_depth'] - 1

    test_mae_list = []
    test_mae_illum_list = []
    test_psnr_list = []
    test_count = 0

    with torch.no_grad():
        model = HDRPointwiseNN(params=model_params)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        # for i, (low_test, full_test, target_test, target_illum_test) in enumerate(tqdm(test_loader)):
        for i, (low_test, full_test, target_test, target_illum_test, fname) in enumerate(tqdm(test_loader)):
            low_test = low_test.to(device)
            full_test = full_test.to(device)
            t_test = target_test.to(device)
            res_test = model(low_test, full_test)
            
            # normalizing to fit the G as same as input's G 
            res_test *= (full_test[:, 1, :, :] / res_test[:, 1, :, :])

            res_test = torch.clamp(res_test, 0, 1)
            t_test = torch.clamp(t_test, 0, 1)

            # transpose imgs since input format should be channel-last for mae function
            res_test_for_mae = res_test.cpu().detach().numpy().transpose(0,2,3,1)
            t_test_for_mae = t_test.cpu().detach().numpy().transpose(0,2,3,1)

            # get illum result, target and masks
            target_illum_test = target_illum_test.numpy()
            full_test_for_illum = full_test.cpu().detach().numpy().transpose(0,2,3,1)
            res_illum_test_for_mae =  (full_test_for_illum) / (res_test_for_mae + 1e-8)
            res_illum_test_for_mae /= (res_illum_test_for_mae[..., 1:2] + 1e-8)
            res_illum_test_for_mae *= full_test_for_illum[..., 1:2]

            mask1 = res_test_for_mae > 1e-8
            mask1 = np.sum(mask1, axis=-1) == 3
            mask2 = full_test_for_illum > 1e-8
            mask2 = np.sum(mask2, axis=-1) == 3
            zero_out_mask = np.logical_or(mask1, mask2)
            
            mae_illum = float(mae(res_illum_test_for_mae, target_illum_test, mask=zero_out_mask))
            test_mae_illum_list.append( mae_illum )
            test_mae_list.append( float(mae(res_test_for_mae, t_test_for_mae)) )
            _psnr = float(psnr(res_test, t_test).item())
            test_psnr_list.append( _psnr )

            test_count += 1

            # img write
            res_illum_test_for_mae /= (res_illum_test_for_mae[..., 1:2] + 1e-8)
            res_illum_test_for_mae *= 0.2
            target_illum_test *= 0.2

            img_t = (t_test.cpu().detach().numpy()).transpose(0,2,3,1)[0] * max_value
            img_res = (res_test.cpu().detach().numpy()).transpose(0,2,3,1)[0] * max_value
            img_orig = full_test_for_illum.squeeze() * max_value 
            res_illum_test_for_mae = res_illum_test_for_mae.squeeze() * max_value
            target_illum_test = target_illum_test.squeeze() * max_value

            # RGB to BGR
            img_t = img_t[:, :, ::-1]
            img_res = img_res[:, :, ::-1]
            img_orig = img_orig[:, :, ::-1]
            res_illum_test_for_mae = res_illum_test_for_mae[:, :, ::-1]
            target_illum_test = target_illum_test[:, :, ::-1]

            # save images
            vis_result = visualize(img_orig, img_res, img_t, data_root='') 
            cv2.imwrite(f'test_result_samples/{fname[0]}.png',vis_result)

    test_mae_list = np.asarray(test_mae_list)
    test_psnr_list = np.asarray(test_psnr_list)

    print("=== Result ===")
    print("mae on illum")
    print("mean:", np.mean(test_mae_illum_list), "\tmedian:", np.median(test_mae_illum_list))
    print("mae on image")
    print("mean:", np.mean(test_mae_list), "\tmedian:", np.median(test_mae_list))
    print("psnr")
    print("mean:", np.mean(test_psnr_list), "\tmedian:", np.median(test_psnr_list))
