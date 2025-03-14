import torch
from termcolor import colored
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from torchkbnufft import (MriSenseNufft,
						  AdjMriSenseNufft,
						  KbNufft,
						  AdjKbNufft,
						  ToepNufft,
						  ToepSenseNufft, KbInterpBack, KbInterpForw)
from pathlib import Path
import warnings
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

from simulation import generate_dyn_shepp_logan
from simulation import add_gaussian_noise
from simulation import magnitude
from simulation import normal_std_from_rician
from simulation import generate_ktraj_rad
from simulation import DynNUFFT

def main(img_dim = [320, 320, 1],
         Na = 503, 
         tindx = 0,
         osmpl_fact = 2,
         noise_mean = 0,
         noise_std = 0.05,
         show_plot_flag = False) -> None:
    
    # Simulation 5
    # To confirm that proposed energy calculation maintains the same values in image and k-space domain
    # General
    Nr = img_dim[0]
    shepp_logan = generate_dyn_shepp_logan(len_with_padding=img_dim[0], len_no_padding=180, Nt=img_dim[2])
    shepp_logan = torch.tensor(shepp_logan, dtype=torch.float).cuda()
    shepp_logan = add_gaussian_noise(shepp_logan, mean=noise_mean , std=noise_std)
    # Calculate image domain power
    shepp_logan_mag = magnitude(shepp_logan)
    shepp_logan_power = (shepp_logan_mag[...,tindx]**2).sum() / (Nr**2)
    # Making k-space trajectories
    krad = (np.linspace(-Nr//2, Nr//2, Nr)/Nr) * (2*np.pi)
    kang = np.linspace(0, Na-1, Na)*(np.pi/180)*(180*0.618034) # Golden angle
    ktraj_stacked = generate_ktraj_rad(krad, kang)
    # K-Space No Coil Map
    NUFFT = DynNUFFT(img_dim=img_dim, osmpl_fact=osmpl_fact, Na=Na, norm='ortho')
    shepp_kdata = NUFFT(shepp_logan, ktraj_stacked)
    # Display kdat and Calculate Noise
    shepp_kdata = torch.reshape(shepp_kdata, (-1, 1, 2, Na, Nr)).permute(1, 2, 3, 4, 0)
    shepp_kdata_mag = magnitude(shepp_kdata)
    # Calculating noise from radial k-space edge
    knoise_mag = torch.cat((shepp_kdata_mag[..., :3, :], shepp_kdata_mag[..., -3:, :]), 1) * (np.sqrt(osmpl_fact)**2)
    std_normal_m, std_normal_s, _, _ = normal_std_from_rician(knoise_mag)
    # Calculating Signal Energy
    kradius = torch.reshape(ktraj_stacked, (-1, 1, 2, Na, Nr)).permute(1, 2, 3, 4, 0)
    kradius = magnitude(kradius)
    adjusted_shepp_kdata_mag = ((shepp_kdata*kradius)**2).sum(2).sqrt() * np.sqrt(osmpl_fact)**2 * np.sqrt(Nr)
    # adjusted_shepp_kdata_mag = torch.reshape(adjusted_shepp_kdata_mag, (Na, Nr))  
    shepp_kdata_energy = (adjusted_shepp_kdata_mag[...,tindx]**2).sum() * (1/Na)
    shepp_kdata_power = shepp_kdata_energy / (Nr**2)
    # Plotting radial k-data
    if show_plot_flag==True:
        img = torch.reshape(adjusted_shepp_kdata_mag, (Na, Nr))
        matplotlib.use('TkAgg')
        plt.figure()
        plt.imshow(img.cpu(), cmap='gray', clim=[0, 1])
        plt.show()        
        matplotlib.use('TkAgg')
        plt.figure()
        plt.imshow(shepp_logan_mag[0,0].cpu(), cmap='gray')
    # Printing results
    print(colored("Proposed power calculation to maintain same values in image and k-space domain", 'magenta'))
    print(colored("Power calculated in the image domain: {}".format(shepp_logan_power), 'white'))
    print(colored("Noise power added in image domain : {}".format(noise_std), 'white'))
    print(colored("Power calculated from radial k-space: {}".format(shepp_kdata_power), 'white'))
    print(colored("Noise power calculated in radial k-space (m) : {}".format(std_normal_m), 'white'))
    print(colored("Noise power calculated in radial k-space (s) : {}".format(std_normal_s), 'white'))

if __name__ == "__main__":
    main()