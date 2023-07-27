import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np
import hdf5storage
import SimpleITK as sitk
import scipy.io as sio
import os

from utils import c2r, x_paired_paths_from_folder, normalize, MaxMinNormalization, Fourier_2D, center_crop

class modl_dataset(Dataset):
    def __init__(self, mode, dataroot_in, dataroot_in_test, dataroot_mask, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.mode = mode
        self.in_folder = dataroot_in
        self.in_folder_test = dataroot_in_test
        self.mask_folder = dataroot_mask
        self.filename_tmpl = '{}'
        self.sigma = sigma

        self.paths = x_paired_paths_from_folder(  # hwgm1
            [self.in_folder, self.mask_folder, self.in_folder_test], ['in', 'mask'],
            self.filename_tmpl, self.mode)  # gmv2

    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """
        in_path_gt = self.paths[index]['in_path']
        x_name = str(os.path.splitext(os.path.basename(in_path_gt))[0])  # 获取文件名
        gt_struct = hdf5storage.loadmat(in_path_gt)
        # gt_struct = sio.loadmat(in_path_gt)  # gmv2
        gt_data = gt_struct['sino'].astype(np.complex64)
        # gt_data = center_crop(gt_data)
        gt = (gt_data / abs(gt_data).max())

        mask_path = self.paths[0]['mask_path']
        # 获取数据集
        mask_data = hdf5storage.loadmat(mask_path)['mask08']  # gmv2
        
        mask = np.fft.fftshift(mask_data)
        mask = mask.astype(np.float32)
        x0 = undersample(gt, mask, self.sigma)
        # mask = np.stack([mask, mask])
        # return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(c2r(csm)), torch.from_numpy(mask)
        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(mask), x_name

    def __len__(self):
        return len(self.paths)

def undersample(gt, mask, sigma):
    """
    :get fully-sampled image, undersample in k-space and convert back to image domain
    """
    nrow, ncol = gt.shape

    sample_idx = np.where(mask.flatten()!=0)[0]
    noise = np.random.randn(len(sample_idx)) + 1j*np.random.randn(len(sample_idx))
    noise = noise * (sigma / np.sqrt(2.))
    b = piA(gt, mask, nrow, ncol) + noise #forward model   piA: fully-sampled image -> undersampled k-space
    atb = piAt(b, mask, nrow, ncol)   # k-space -> zero-filled reconstruction
    return atb

def piA(im, mask, nrow, ncol):
    """
    fully-sampled image -> undersampled k-space
    """
    im = np.reshape(im, (nrow, ncol))
    im_coil = np.tile(im, [1, 1]) #split coil images
    k_full = np.fft.fft2(im_coil, norm='ortho') #fft
    if len(mask.shape) == 2:
        mask = np.tile(mask, (1, 1))
    k_u = k_full[mask!=0]
    return k_u

def piAt(b, mask, nrow, ncol):
    """
    k-space -> zero-filled reconstruction
    """
    if len(mask.shape) == 2:
        mask = np.tile(mask, (1, 1))
    zero_filled = np.zeros((nrow, ncol), dtype=np.complex64)
    zero_filled[mask!=0] = b #zero-filling
    img = np.fft.ifft2(zero_filled, norm='ortho') #ifft
    coil_combine = img.astype(np.complex64) #coil combine
    return coil_combine