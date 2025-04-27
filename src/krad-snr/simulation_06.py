import warnings
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.gridspec as gridspec

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
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
import os

from simulation import (
    DynSENSE,
    add_gaussian_noise,
    generate_dyn_shepp_logan,
    generate_ktraj_rad,
    magnitude,
    normal_std_from_rician,
)
from tqdm import tqdm

# import torch.nn as nn

# import argparse
# from PIL import Image
# from PIL import ImageDraw
# import imageio
# import random
# from PIL import Image

# import scipy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_simulation(img_dim = [320, 320, 3],
         Na = 503, 
         tindx = 0,
         osmpl_fact = 5,
         noise_mean = 0.0,
         noise_std = 0.001,
         nbootstrap = 1,
         verbose = True,
         show_plot_flag = False,
         device = 'cuda') -> None:
    
    # General Initializations
    
    
    ksignal_area_fraction = 0.90
    rlim = int(img_dim[1]-(ksignal_area_fraction*img_dim[1]))
    Nt = img_dim[2]

    # Simulated coil maps
    ncoils = 32
    csmap_dim = [1, ncoils, 2, img_dim[0], img_dim[1]]
    csmap = torch.ones(csmap_dim, dtype = torch.float, device=device)/(np.sqrt(2*32))
    # csmap_path = '/data/brahma01/Datasets/Uncompressed_coils.npy'
    # csmap = torch.tensor(np.load(csmap_path), dtype = torch.float, device=device)

    # We assumed that each coil have identical noise power. Thus, noise
    # power (and consequently the standard deviation) in each coil is given
    # by the following:
    total_noise_power = noise_std**2
    noise_std = np.sqrt(total_noise_power/ncoils)
    
    # Simulation 6
    # For the case of k-space shepp logan with invivo Multicoil
    # check that calculated SNR corresponds to the original SNR
    # General
    Nr = img_dim[0]
    shepp_logan = 10*generate_dyn_shepp_logan(len_with_padding=img_dim[0], len_no_padding=180, Nt=img_dim[2]) 
    shepp_logan = torch.tensor(shepp_logan, dtype=torch.float, device=device)

    # Adding noise
    noise_std_tensor = noise_std * torch.ones(shepp_logan.shape)
    shepp_logan = add_gaussian_noise(shepp_logan, mean=noise_mean, std=noise_std_tensor)

    # Calculate image domain power
    shepp_logan_mag = magnitude(shepp_logan)
    shepp_logan_energy = (shepp_logan_mag[...]**2).sum(dim=(0,1,2))
    shepp_logan_power = shepp_logan_energy / (Nr**2)
    
    # Making k-space trajectories
    krad = (np.linspace(-Nr//2, Nr//2, Nr)/Nr) * (2*np.pi)
    kang = np.linspace(0, Na-1, Na)*(np.pi/180)*(180*0.618034) # Golden angle
    ktraj_stacked = generate_ktraj_rad(krad, kang) # .unsqueeze(-1).repeat(1, 1, 1, Nt)
    
    # Applying SENSE forward operator and add noise
    SENSE = DynSENSE(csmap=csmap, img_dim=img_dim, osmpl_fact=osmpl_fact, Na=Na, norm='ortho')
    shepp_kdata_no_noise = SENSE(shepp_logan, ktraj_stacked) * np.sqrt(osmpl_fact)**2
    
    # Bootstrapping
    SNR_boot = torch.zeros((nbootstrap,Nt))
    for nboot in tqdm(range(nbootstrap)):
        
        # Noise simulation
        noise_std_tensor = noise_std * torch.ones(shepp_kdata_no_noise.shape)
        shepp_kdata = add_gaussian_noise(shepp_kdata_no_noise.clone(), mean=noise_mean, std=noise_std_tensor) # shepp_kdata_no_noise.clone() #
    
        # Display kdat and Calculate Noise
        Nc = csmap.shape[1]
        # shepp_kdata = torch.reshape(shepp_kdata, (img_dim[2], Nc, 2, Na, Nr)).permute(1, 2, 3, 4, 0)

        shepp_kdata = rearrange(shepp_kdata, 'nt ncoil ncplx (Na Nr) -> 1 ncplx ncoil Na Nr nt', Na=Na)
    
        # Calculating noise from radial k-space edge
        knoise = shepp_kdata.clone()
        knoise = torch.cat((knoise[..., :3, :], knoise[..., -3:, :]), -2)
        knoise_std = knoise.std((0,1,3,4))
        knoise_power = (knoise_std**2).sum((0))
        
        # Ramp filter response
        kradius = rearrange(ktraj_stacked, '1  ncplx (Na Nr) -> 1 ncplx 1 Na Nr 1', Na=Na)
        kradius = magnitude(kradius).unsqueeze(1).sqrt()
        
        # Calculating compensation term
        npcom = 100 # 5 # 
        pcomp = torch.zeros([Nc, Nt], device=shepp_kdata.device)
        noise_std_tensor = rearrange(knoise_power.sqrt()/ncoils, 'nt -> 1 1 1 1 1 nt') * torch.ones(shepp_kdata.shape, device=knoise_power.device)
        for _ in range(npcom):
            knoise_comp = add_gaussian_noise(
                                torch.zeros(
                                        shepp_kdata.shape, device=shepp_kdata.device
                                                ), mean=noise_mean, std=noise_std_tensor)
            knoise_comp[..., :rlim, :], knoise_comp[..., -rlim:, :] = 0, 0
            knoise_comp_ramp_mag = (magnitude(knoise_comp*kradius))
            pcomp += ((knoise_comp_ramp_mag**2).sum((0,2,3)) * (1/(2*Na)))
        pcomp = pcomp/npcom
        
        # Calculating signal power from radial k-space core region
        ksignal = shepp_kdata.clone()
        ksignal[..., :rlim, :], ksignal[..., -rlim:, :] = 0, 0
        ksignal_ramp_mag = (magnitude(ksignal*kradius)) # (magnitude(kradius)) # 
        # Compatibility with 1D FFT normalization
        # For 2D it is Nr x Nr, whereas for 1D it should only be Nr

        ksignal_energy_coil = (((ksignal_ramp_mag**2).sum((0,2,3))
                                 * (1/(2*Na)))
                                   - pcomp) * Nr

        # ksignal_energy_coil = (ksignal_ramp_mag**2).sum((0,2,3))
        ksignal_power_coil = ksignal_energy_coil/(Nr**2)
        ksignal_power = ksignal_power_coil.sum((0))

        # print('ksignal power: ' + str(ksignal_power))
        
        # SNR values
        original_noise_power = total_noise_power
        SNR_gnd = 10*torch.log10(shepp_logan_power/original_noise_power)
        SNR_boot[nboot] = 10*torch.log10(ksignal_power/knoise_power)
        
    SNR_est = SNR_boot.mean(axis=0)

    if torch.isnan(SNR_est).any():
        print('Using nanmean')
        SNR_est = np.nanmean(SNR_boot, axis=0)
    
    # Printing results
    if verbose == True:
        print(colored("Proposed power calculation to maintain same values in image and k-space domain", 'magenta'))
        print(colored("Power calculated in the image domain: {}".format(shepp_logan_power), 'white'))
        print(colored("Noise power added in k-space domain : {}".format(original_noise_power), 'white'))
        print(colored("Power calculated from radial k-space: {}".format(ksignal_power), 'white'))
        print(colored("Noise power calculated in radial k-space (m) : {}".format(knoise_power), 'white'))
        # print(colored("Noise power calculated in radial k-space (s) : {}".format(knoise_power_s), 'white'))
        print(colored("Signal-to-noise ratio values", 'magenta'))
        print(colored("Original SNR value: {}".format(SNR_gnd), 'white'))
        print(colored("Estimated SNR value : {}".format(SNR_est), 'white'))
        
    return SNR_boot, SNR_gnd

def create_cfg() -> Dict:

    # Creating dictionary
    cfg = {}
    cfg["Nr"] = 320
    cfg["Nt"] = 1
    cfg["nspokes_array"] = np.array([503, 251, 75]) # np.array([503, 251, 75, 50, 36]) #
    cfg["noise_std_list"] = [1.0, 0.8, 0.5, 0.2, 0.1, 0.05] #  [0.8, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005] # 
    cfg["nbootstrap"] = 5 # 100
    cfg["device"] = 'cuda'

    return cfg

def run_simulation_batch(cfg) -> None:

    # Unpacking configuration
    Nr = cfg["Nr"]
    Nt = cfg["Nt"]
    nspokes_array = cfg["nspokes_array"]
    noise_std_list = cfg["noise_std_list"]
    nbootstrap = cfg["nbootstrap"]
    device = cfg["device"]

    # Constructing place holders
    SNR_est_array = np.zeros([nspokes_array.__len__(), noise_std_list.__len__(), nbootstrap, Nt])
    SNR_gnd_array = np.zeros([nspokes_array.__len__(), noise_std_list.__len__(), Nt])
    
    for i, nspokes in enumerate(nspokes_array):
        for j, noise_std in enumerate(noise_std_list):
            SNR_est, SNR_gnd = run_simulation(img_dim = [Nr, Nr, Nt], 
                                    Na = nspokes, 
                                    noise_std = noise_std,
                                    nbootstrap = nbootstrap,
                                    verbose=True,
                                    device=device)
            # Print Results
            print('noise_std: '+ str(noise_std))
            print('SNR_est: '+ str(SNR_est.mean()))
            print('SNR_gnd: '+ str(SNR_gnd))
            
            # Store Results
            SNR_est_array[i, j] = SNR_est.cpu()
            SNR_gnd_array[i, j] = SNR_gnd.cpu()
            
    # Save arrays
    save_path = str(Path(__file__).resolve().parent.parent.parent / "results")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(Path.joinpath(Path(save_path), "nspokes.npy"), nspokes_array)
    np.save(Path.joinpath(Path(save_path), "SNR_est.npy"), SNR_est_array)
    np.save(Path.joinpath(Path(save_path), "SNR_gnd.npy"), SNR_gnd_array)

def plot_results(cfg) -> None:

    # Unpacking configuration
    Nr = cfg["Nr"]

    # Loading arrays
    save_path = str(Path(__file__).resolve().parent.parent.parent / "results")
    nspokes_array = np.load(Path.joinpath(Path(save_path),'nspokes.npy'))
    SNR_est_array = np.load(Path.joinpath(Path(save_path),'SNR_est.npy'))
    SNR_gnd_array = np.load(Path.joinpath(Path(save_path),'SNR_gnd.npy'))
    
    # Color map
    cmap = matplotlib.cm.get_cmap('viridis_r')
    colors = cmap(np.linspace(0, 1, SNR_gnd_array.shape[0]))
    nspokes_max = int(np.ceil((np.pi/2)*Nr))
    R_list = [int(i) for i in np.round(nspokes_max/nspokes_array)]
    
    # Plotting
    # matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(5,5))
    plt.title("Estimated SNR")
    for i, color in enumerate(colors):
        label = '$R$: ' + str(R_list[i])
        # if R_list[i] in [1,7]:
        #     continue
        if i==0:
            plt.plot(SNR_gnd_array[i].reshape(-1), SNR_gnd_array[i].reshape(-1), linewidth=1, linestyle="dashed", label='Perfect estimation', color='black')        
        plt.plot(SNR_gnd_array[i].reshape(-1), np.nanmean(SNR_est_array[i], axis=-2).reshape(-1), linewidth=1, label=label, color=color)
        plt.fill_between(SNR_gnd_array[i].reshape(-1), 
                         np.nanmean(SNR_est_array[i], axis=-2).reshape(-1) - np.nanstd(SNR_est_array[i], axis=-2).reshape(-1), 
                         np.nanmean(SNR_est_array[i], axis=-2).reshape(-1) + np.nanstd(SNR_est_array[i], axis=-2).reshape(-1), linewidth=1, alpha=0.2, color=color)
    plt.xlabel('$SNR_{\mathrm{GND}}$')
    plt.ylabel('$SNR_{\mathrm{EST}}$')
    plt.legend(loc="upper left")
    # # Zoomed in figure
    # plt.ylim([3, 8])
    # plt.xlim([4, 8])
    # plt.xticks([4, 6, 8])
    # plt.yticks([3, 5, 7])
    # plt.tick_params(axis='both', which='major', labelsize=20)
    # Save figure
    plt.grid()
    plt.show()
    fig.savefig(Path.joinpath(Path(save_path), 'SNR_plot'), dpi=500)

def main() -> None:

    cfg = create_cfg()

    run_simulation_batch(cfg)

    plot_results(cfg)
    
if __name__ == "__main__":
    main()
    
    