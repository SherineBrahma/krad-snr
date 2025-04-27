import warnings
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from termcolor import colored
from torchkbnufft import (
    AdjKbNufft,
    AdjMriSenseNufft,
    KbInterpBack,
    KbInterpForw,
    KbNufft,
    MriSenseNufft,
    ToepNufft,
    ToepSenseNufft,
)

warnings.filterwarnings("ignore")

# import torch.nn as nn

# import argparse
# from PIL import Image
# from PIL import ImageDraw
# import imageio
# import random
# from PIL import Image

# import scipy

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def fftshift(x):

    shift_x = x.shape[2] // 2
    shift_y = x.shape[3] // 2
    x = torch.roll(x, shifts=(shift_x, shift_y), dims=(2, 3))

    return x

def generate_dyn_shepp_logan(len_with_padding = 320, len_no_padding = 180, Nt = 1):
    # Generate shepp-logan phantom    
    shepp_logan = shepp_logan_phantom()
    shepp_logan = rescale(shepp_logan, scale=len_no_padding/shepp_logan.shape[0], mode='reflect')
    pad_width = int((len_with_padding-len_no_padding)/2)
    shepp_logan = np.pad(shepp_logan, [(pad_width, pad_width), (pad_width, pad_width)], mode='constant', constant_values=0)
    # Simulate dynamic image from shepp-logan phantom
    shepp_logan = torch.tensor(shepp_logan[None, None, :, :])
    shepp_logan = torch.stack(Nt * [shepp_logan],dim=-1)
    shepp_logan = torch.cat((shepp_logan, torch.zeros(shepp_logan.shape)), 1)
    
    return shepp_logan

def add_gaussian_noise(xin, mean = 0, std = 0.01):    
    # Generate and add noise
    noise = torch.normal(mean=mean, std=std).to(xin.device)
    xout = xin + noise

    # # Generate and add noise
    # noise = torch.empty(xin.shape).normal_(mean=mean, std=std).to(xin.device)
    # xout = xin + noise
    
    return xout

def magnitude(xin):    
    # Calculate magnitude
    xout = (xin**2).sum(1).sqrt()
    
    return xout

def normal_std_from_rician(xin):    
    # Calculate normal standard deviation from 
    # rician mean and standard deviation
    mean_rician = torch.mean(xin)
    std_rician = torch.std(xin)
    pi = torch.tensor(np.pi)
    std_normal_m = torch.sqrt(2/pi) * mean_rician
    std_normal_s = torch.sqrt(2/(4-pi)) * std_rician
    
    return std_normal_m, std_normal_s, mean_rician, std_rician

def fft(xin):    
    # Calculate FFT
    xfft = xin.permute(0, 4, 2, 3, 1)
    xfft = torch.Tensor.fft(xfft, 2, normalized=True)
    xfft = xfft.permute(0, 4, 2, 3, 1)
    xfft = xfft[:, 0, :] + (1j * xfft[:, 1, :])
    xout = torch.cat((fftshift(xfft[:, None, :].real), fftshift(xfft[:, None, :].imag)), 1)
    
    return xout

def extract_corner_patch(xin, patch_height=10, patch_width=10):
    # Extracting patch of specified height and width
    _, _, nx, ny, _ = xin.shape
    height_min, height_max = ny-patch_height, ny
    width_min, width_max = nx-patch_width, nx
    patch = xin[:, :, width_min:width_max, height_min:height_max ,:]
    
    return patch

def generate_kdcf(Nr, Na, show_plot_flag=False):
    # Making density compensation function
    del_kr = (2*np.pi)/Nr
    kdcf = np.zeros(Nr)
    kdcf[Nr//2] = (del_kr**3)/8
    for i in range(1,Nr//2):
        kdcf[Nr//2 + i -1]= ((2*np.pi)/(Nr-1)) * del_kr**2 * i
        kdcf[Nr//2 - i -1] = ((2*np.pi)/(Nr-1)) * del_kr**2 * i
    kdcf[kdcf.__len__()-1] = ((2*np.pi)/(Nr-1)) * del_kr**2 * (Nr//2)
    kdcf = torch.tensor(np.tile(kdcf, Na), dtype=torch.float)
    kdcf = torch.cat((kdcf[None, None, None, :], kdcf[None, None, None, :]), 2)    
    # Plotting density-compensation function
    if show_plot_flag==True:
        matplotlib.use('TkAgg')
        plt.figure()
        plt.plot(kdcf[0,0,0,1:2000], linewidth=1)
        plt.xlabel('1st Spoke')
        plt.ylabel('Profile') 
        plt.show()
    
    return kdcf

def generate_ktraj_rad(krad, kang):
    # Making k-space trajectories
    Nr = krad.shape[0]
    Na = kang.shape[0]
    ktraj = np.zeros((Nr, Na, 2), dtype=np.float32)
    # Making the x and y coordinates
    ktraj[:, :, 0] = np.dot(krad[:,np.newaxis], np.cos(kang[np.newaxis,:]))
    ktraj[:, :, 1] = np.dot(krad[:,np.newaxis], np.sin(kang[np.newaxis,:]))
    # Our version of kbNUFFT expects the dimension of ktraj as [2, Nrad] where Nrad=Ny*Na
    ktraj = torch.tensor(ktraj)
    ktraj = ktraj.permute(2, 0, 1)
    ktraj_stacked = ktraj[...,0]
    for aindx in range(1,Na):
        ktraj_stacked = torch.cat( (ktraj_stacked,ktraj[...,aindx]), dim=1)
    ktraj_stacked = torch.tensor(ktraj_stacked[None, :], dtype=torch.float).cuda()
    
    return ktraj_stacked        
        
def radial_NUFFT(Nr, Na, krad):
    # Making k-space trajectories
    kang = np.linspace(0, Na-1, Na)*(np.pi/180)*(180*0.618034) # Golden angle
    ktraj = np.zeros((Nr, Na, 2), dtype=np.float32)
    # Making the x and y coordinates
    ktraj[:, :, 0] = np.dot(krad[:,np.newaxis], np.cos(kang[np.newaxis,:]))
    ktraj[:, :, 1] = np.dot(krad[:,np.newaxis], np.sin(kang[np.newaxis,:]))
    # Our version of kbNUFFT expects the dimension of ktraj as [2, Nrad] where Nrad=2*Ny*n_spokes
    ktraj = torch.tensor(ktraj)
    ktraj = ktraj.permute(2, 0, 1)
    ktraj_stacked = ktraj[...,0]
    for aindx in range(1,Na):
        ktraj_stacked = torch.cat( (ktraj_stacked,ktraj[...,aindx]), dim=1)
    ktraj_stacked = torch.tensor(ktraj_stacked[None, :], dtype=torch.float).cuda()
    
    return ktraj_stacked

class DynNUFFT():

    def __init__(self, img_dim, osmpl_fact, Na, norm='ortho'):
        
        Nr = img_dim[0]
        grid_size = [osmpl_fact*Nr, osmpl_fact*Nr]        
        self.NUFFT = KbNufft(im_size=[img_dim[0], img_dim[1]], grid_size=grid_size, norm=norm).to(torch.float).cuda()
        self.vec_spokes_len = Na * Nr
        
    def __call__(self, xin, ktraj):
        
        # Calculate dynamic 2D kdata
        xnufft = xin.permute(4, 0, 1, 2, 3)
        kdata = torch.zeros([xnufft.shape[0], xnufft.shape[1], xnufft.shape[2], self.vec_spokes_len], device=xin.device)
        for i in range(xin.shape[0]):
            kdata[i] = self.NUFFT(xnufft[i:i+1], ktraj)
        
        return kdata
    
class DynAdjKbNUFFT():

    def __init__(self, img_dim, osmpl_fact, Na, norm='ortho'):
        
        self.Nr = img_dim[0]
        grid_size = [osmpl_fact*self.Nr, osmpl_fact*self.Nr]        
        self.AdjNUFFT = AdjKbNufft(im_size=[img_dim[0], img_dim[1]], grid_size=grid_size, norm='ortho').to(torch.float).cuda()
        
    def __call__(self, xin, ktraj):
        
        # Calculate dynamic 2D image
        xout = torch.zeros([xin.shape[0], xin.shape[1], xin.shape[2], self.Nr, self.Nr])
        for i in range(xin.shape[0]):
            xout[i] = self.AdjNUFFT(xin[i:i+1], ktraj)
        xout = xout.permute(1, 2, 3, 4, 0)
        
        return xout
    
class DynSENSE():

    def __init__(self, csmap, img_dim, osmpl_fact, Na, norm='ortho'):
        
        Nr = img_dim[0]
        grid_size = [osmpl_fact*Nr, osmpl_fact*Nr]        
        self.SENSE = MriSenseNufft(smap=csmap, im_size=[img_dim[0], img_dim[1]], grid_size=grid_size, norm='ortho').to(torch.float).cuda()
        self.ncoils = csmap.shape[1]
        self.vec_spokes_len = Na * Nr
        
    def __call__(self, xin, ktraj):
        
        # Calculate dynamic 2D kdata
        xnufft = xin.permute(4, 0, 1, 2, 3)
        kdata = torch.zeros([xnufft.shape[0], self.ncoils, xnufft.shape[2], self.vec_spokes_len], device=xin.device)
        for i in range(xnufft.shape[0]):
            kdata[i] = self.SENSE(xnufft[i:i+1], ktraj)
        
        return kdata