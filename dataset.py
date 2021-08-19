import os
import numpy as np
from PIL import Image
import cv2
from time import time
import json

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class HDRDataset(Dataset):
    def __init__(self, image_path, params=None, suffix='', mode='train'):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.image_path = image_path
        self.suffix = suffix
        in_files = self.list_files(os.path.join(image_path, mode))
        self.in_files = sorted([f for f in in_files if f.split('/')[-1].count('_') == 2])
        
        self.max = 2 ** params['bit_depth'] - 1
        self.ls = params['net_input_size']
        self.fs = params['net_output_size']
        self.low = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.full = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        fname = os.path.split(self.in_files[idx])[-1]
        path = self.in_files[idx].replace(fname, '')
        fname_split = fname.split('.')
        fname = fname_split[0]
        extension = fname_split[1]

        imagein = cv2.imread(self.in_files[idx], cv2.IMREAD_UNCHANGED)
        imagein = cv2.cvtColor(imagein, cv2.COLOR_BGR2RGB).astype('float32') / self.max

        imagein_low = cv2.resize(imagein, (self.ls, self.ls))
        
        imageout = cv2.imread(path + fname + '_gt.' + extension, cv2.IMREAD_UNCHANGED)
        imageout = cv2.cvtColor(imageout, cv2.COLOR_BGR2RGB).astype('float32') / self.max

        imagein_low = self.low(imagein_low)
        imagein_full = self.full(imagein)
        imageout = self.full(imageout)
        
        gt_illum = np.load(path + fname + '_illum.npy').astype('float32')
        
        return imagein_low, imagein_full, imageout, gt_illum, fname

    def list_files(self, in_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            files.extend(filenames)
            break
        files = sorted([os.path.join(in_path, x) for x in files])
        return files