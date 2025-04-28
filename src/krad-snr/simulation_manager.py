from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from krad_snr import krad_snr
from termcolor import colored
from tqdm import tqdm
from utils import (
    DynSENSE,
    add_gaussian_noise,
    generate_dyn_shepp_logan,
    generate_ktraj_rad,
    magnitude,
)


def run_simulation(
        img,
        MROpt,
        ktraj,
        nspokes,
        noise_mean=0.0,
        noise_std=0.001,
        nbootstrap=1,
        verbose=True) -> None:
    """
    Perform SNR simulation for the given set of parameters.

    Args:
        img (torch.Tensor): Input dynamic image (dim: nb, ncplx, nx, ny, nt).
                            Here, nb is batch size, ncplx is number of
                            complex channels, nx and ny are image dimensions,
                            and nt is the number of time frames.
        MROpt (DynSENSE): Forward operator (SENSE-based).
        ktraj (np.ndarray): k-space trajectory.
        nspokes (int): Number of spokes per frame.
        noise_mean (float): Mean of Gaussian noise to add.
        noise_std (float): Standard deviation of Gaussian noise to add.
        nbootstrap (int): Number of bootstrap trials.
        verbose (bool): Whether to print detailed results.

    Returns:
        SNR_boot (torch.Tensor): Bootstrapped SNR estimates (nbootstrap x nt).
        SNR_gnd (torch.Tensor): Ground-truth SNR values (nt,).
    """

    # General Initializations
    _, _, nx, ny, nt = img.shape
    ncoils = MROpt.ncoils

    # Compute per-coil noise standard deviation
    # We assumed that each coil have identical noise power. Thus, noise
    # power (and consequently the standard deviation) in each coil is given
    # by the following:
    total_noise_power = noise_std**2
    noise_std = np.sqrt(total_noise_power/ncoils)

    # Compute image-domain signal power
    img_mag = magnitude(img)
    img_energy = (
        img_mag[...]**2
        ).sum(dim=(0, 1, 2))
    img_power = img_energy / (nx*ny)

    # Add noise to image
    noise_std_tensor = noise_std * torch.ones(img.shape)
    img = add_gaussian_noise(
        img, mean=noise_mean,
        std=noise_std_tensor
        )

    # Applying SENSE forward operator to get k-space data
    kdata_no_noise = MROpt(
        img, ktraj
        ) * np.sqrt(MROpt.osmpl_fact)**2
    
    # Bootstrapping SNR estimation
    SNR_boot = torch.zeros((nbootstrap, nt))
    ktraj = rearrange(ktraj,
                      '1  ncplx (nspokes nr) -> 1 ncplx 1 nspokes nr 1',
                      nspokes=nspokes)
    for nboot in tqdm(range(nbootstrap)):

        # Add noise to k-space data
        noise_std_tensor = noise_std * torch.ones(kdata_no_noise.shape)
        kdata = add_gaussian_noise(kdata_no_noise.clone(),
                                   mean=noise_mean,
                                   std=noise_std_tensor)

        # Calculate SNR using krad tecnhique
        kdata = rearrange(
            kdata,
            'nt ncoil ncplx (nspokes nr) -> 1 ncplx ncoil nspokes nr nt',
            nspokes=nspokes
            )
        snr, ksignal_power, knoise_power = krad_snr(kdata,
                                                    ktraj,
                                                    noise_mean)

        # Store SNR
        original_noise_power = total_noise_power
        SNR_gnd = 10*torch.log10(img_power/original_noise_power)
        SNR_boot[nboot] = snr

    # Aggregate bootstrapped SNR estimates
    SNR_est = SNR_boot.mean(axis=0)
    if torch.isnan(SNR_est).any():
        print('Using nanmean')
        SNR_est = np.nanmean(SNR_boot, axis=0)

    # Print summary
    if verbose:
        print(
            colored(
                ("Proposed power calculation to maintain same values"
                 " in image and k-space domain"),
                'magenta')
            )
        print(
            colored(
                ("Power calculated in the image"
                 " domain: {}".format(img_power)),
                'white')
            )
        print(
            colored(
                ("Noise power added in k-space"
                 " domain : {}".format(original_noise_power)),
                'white')
            )
        print(
            colored(
                ("Power calculated from radial"
                 " k-space: {}".format(ksignal_power)),
                'white')
            )
        print(
            colored(
                ("Noise power calculated in radial"
                 " k-space (m) : {}".format(knoise_power)),
                'white')
            )
        print(
            colored(("Signal-to-noise ratio values"),
                    'magenta')
            )
        print(
            colored(("Original SNR value: {}".format(SNR_gnd)),
                    'white')
            )
        print(
            colored(("Estimated SNR value : {}".format(SNR_est)),
                    'white')
            )

    return SNR_boot, SNR_gnd


def run_simulation_batch(cfg) -> None:
    """
    Run a batch of SNR simulations across different number of spokes and
    noise settings.

    Args:
        cfg (dict): Configuration dictionary specifying simulation parameters.
    """

    # Unpacking configuration
    nr = cfg["nr"]
    nt = cfg["nt"]
    nspokes_array = cfg["nspokes_array"]
    noise_std_list = cfg["noise_std_list"]
    nbootstrap = cfg["nbootstrap"]
    device = cfg["device"]
    osmpl_fact = cfg["osmpl_fact"]

    # Constructing place holders
    SNR_est_array = np.zeros([nspokes_array.__len__(),
                              noise_std_list.__len__(),
                              nbootstrap,
                              nt])
    SNR_gnd_array = np.zeros([nspokes_array.__len__(),
                              noise_std_list.__len__(),
                              nt])

    # Generating shepp-logan phantom
    shepp_logan = 10*generate_dyn_shepp_logan(len_with_padding=nr,
                                              len_no_padding=180,
                                              nt=nt) 
    shepp_logan = torch.tensor(shepp_logan,
                               dtype=torch.float,
                               device=device)

    # Loading coil maps
    # Here, we are using a dummy coil map normalized appropriately.
    # In practice, you should load the coil maps corresponding to your
    # specific MR image
    ncoils = 32
    csmap_dim = [1, ncoils, 2, nr, nr]
    csmap = torch.ones(
        csmap_dim, dtype=torch.float, device=device
        )/(np.sqrt(2*32))

    # Loop over all configurations
    for i, nspokes in enumerate(nspokes_array):
        for j, noise_std in enumerate(noise_std_list):
            
            # Constructing MR Operator
            SENSE = DynSENSE(csmap=csmap,
                             img_dim=[nr, nr, nt],
                             osmpl_fact=osmpl_fact,
                             nspokes=nspokes,
                             norm='ortho')

            # Making k-space trajectories
            krad = (np.linspace(-nr//2, nr//2, nr)/nr) * (2*np.pi)
            kang = (np.linspace(0, nspokes-1, nspokes)
                    * (np.pi/180) *
                    (180*0.618034))
            ktraj = generate_ktraj_rad(krad, kang)

            # Run single simulation
            SNR_est, SNR_gnd = run_simulation(shepp_logan,
                                              SENSE,
                                              ktraj,
                                              nspokes=nspokes,
                                              noise_std=noise_std,
                                              nbootstrap=nbootstrap,
                                              verbose=True)
            # Print Results
            print('noise_std: ' + str(noise_std))
            print('SNR_est: ' + str(SNR_est.mean()))
            print('SNR_gnd: ' + str(SNR_gnd))

            # Store Results
            SNR_est_array[i, j] = SNR_est.cpu()
            SNR_gnd_array[i, j] = SNR_gnd.cpu()

    # Save arrays
    save_path = str(Path(__file__).resolve().parent.parent.parent / "results")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(Path.joinpath(Path(save_path), "nspokes.npy"), nspokes_array)
    np.save(Path.joinpath(Path(save_path), "SNR_est.npy"), SNR_est_array)
    np.save(Path.joinpath(Path(save_path), "SNR_gnd.npy"), SNR_gnd_array)
