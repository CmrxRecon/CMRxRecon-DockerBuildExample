import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np
import hdf5storage
import scipy.io as sio
import os

from utils import c2r, x_paired_paths_from_folder

class modl_dataset(Dataset):
    def __init__(self, mode, dataroot_in, dataroot_in_test, dataroot_in_csm, dataroot_in_csm_test, dataroot_mask, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.mode = mode
        self.in_folder = dataroot_in
        self.in_folder_test = dataroot_in_test
        self.mask_folder = dataroot_mask
        self.csm_folder = dataroot_in_csm
        self.csm_folder_test = dataroot_in_csm_test
        self.filename_tmpl = '{}'
        self.sigma = sigma
        self.paths = x_paired_paths_from_folder(  # hwgm1
            [self.in_folder, self.mask_folder, self.in_folder_test, self.csm_folder,self.csm_folder_test], ['in', 'mask', 'csm'],
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
        gt_data = hdf5storage.loadmat(in_path_gt)['sino'].astype(np.complex64)  # gmv2
        gt = (gt_data / abs(gt_data).max())
        # import matplotlib.pyplot as plt
        # plt.imshow(abs(gt), 'gray')
        # plt.show()
        pre_csm_path = self.paths[index]['csm_path']
        csm_data = hdf5storage.loadmat(pre_csm_path)['SNS'].astype(np.complex64)  # gmv2
        csm_data = np.transpose(csm_data, (2, 0, 1))
        csm = (csm_data / abs(csm_data).max())

        mask_path = self.paths[0]['mask_path']
        mask_data = hdf5storage.loadmat(mask_path)['mask10']  # gmv2
        mask = np.fft.fftshift(mask_data)
        mask = mask.astype(np.float32)

        x0 = undersample(gt, csm, mask, self.sigma)
        # mask = np.stack([mask, mask])
        # return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(c2r(csm)), torch.from_numpy(mask)
        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(mask), x_name

    def __len__(self):
        return len(self.paths)

def undersample(gt, csm, mask, sigma):
    """
    :get fully-sampled image, undersample in k-space and convert back to image domain
    """
    ncoil, nrow, ncol = csm.shape
    sample_idx = np.where(mask.flatten()!=0)[0]
    noise = np.random.randn(len(sample_idx)*ncoil) + 1j*np.random.randn(len(sample_idx)*ncoil)
    noise = noise * (sigma / np.sqrt(2.))
    b = piA(gt, csm, mask, nrow, ncol, ncoil) + noise #forward model
    atb = piAt(b, csm, mask, nrow, ncol, ncoil)
    return atb

def piA(im, csm, mask, nrow, ncol, ncoil):
    """
    fully-sampled image -> undersampled k-space
    """
    im = np.reshape(im, (nrow, ncol))
    im_coil = np.tile(im, [ncoil, 1, 1]) * csm #split coil images
    k_full = np.fft.fft2(im_coil, norm='ortho') #fft
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    k_u = k_full[mask!=0]
    return k_u

def piAt(b, csm, mask, nrow, ncol, ncoil):
    """
    k-space -> zero-filled reconstruction
    """
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    zero_filled = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    zero_filled[mask!=0] = b #zero-filling
    img = np.fft.ifft2(zero_filled, norm='ortho') #ifft
    coil_combine = np.sum(img*csm.conj(), axis=0).astype(np.complex64) #coil combine
    return coil_combine