import torch
from einops import rearrange
from utils import add_gaussian_noise, magnitude


def krad_snr(shepp_kdata, ktraj_stacked, noise_mean, npcom=100):
    """
    Estimate SNR from radial k-space data wher the noise power is
    estimated from the edges of the k-space spokes.

    Args:
        shepp_kdata (torch.Tensor): Radial k-space data with shape
                                    (1, ncplx=2, ncoils, nspokes, nr, nt).
        ktraj_stacked (torch.Tensor): Stacked radial k-space trajectory
                                      (1, ncplx=2, 1, nspokes, nr, 1).
        noise_mean (float): Mean of Gaussian noise (typically 0.0).
        npcom (int, optional): Number of simulations for compensation term
                               estimation. Defaults to 100.

    Returns:
        snr (torch.Tensor): Estimated SNR for each frame (nt,).
        ksignal_power (torch.Tensor): Estimated signal power from the central
                                      region of radial k-space (nt,).
        knoise_power (torch.Tensor): Estimated noise power from the edges
                                     of radial k-space (nt,).
    """

    # General Initializations
    _, _, ncoils, nspokes, nr, nt = shepp_kdata.shape
    ksignal_area_fraction = 0.90
    rlim = int(nr-(ksignal_area_fraction*nr))
    
    # Estimate noise from the outer edges of radial k-space
    knoise = shepp_kdata.clone()
    knoise = torch.cat((knoise[..., :3, :], knoise[..., -3:, :]), -2)
    knoise_std = knoise.std((0, 1, 3, 4))
    knoise_power = (knoise_std**2).sum((0))
    
    # Calculate ramp filter response 
    kradius = magnitude(ktraj_stacked).unsqueeze(1).sqrt()
    
    # Compensation term estimation
    pcomp = torch.zeros([ncoils, nt], device=shepp_kdata.device)
    noise_std_tensor = rearrange(
        knoise_power.sqrt()/ncoils, 'nt -> 1 1 1 1 1 nt'
        ) * torch.ones(shepp_kdata.shape, device=knoise_power.device)
    for _ in range(npcom):
        knoise_comp = add_gaussian_noise(
                            torch.zeros(
                                    shepp_kdata.shape,
                                    device=shepp_kdata.device
                                            ),
                            mean=noise_mean,
                            std=noise_std_tensor
                                            )
        knoise_comp[..., :rlim, :], knoise_comp[..., -rlim:, :] = 0, 0
        knoise_comp_ramp_mag = (magnitude(knoise_comp*kradius))
        pcomp += ((knoise_comp_ramp_mag**2).sum((0, 2, 3)) * (1/(2*nspokes)))
    pcomp = pcomp/npcom

    # Extract central region for signal (zero out edges)
    ksignal = shepp_kdata.clone()
    ksignal[..., :rlim, :], ksignal[..., -rlim:, :] = 0, 0

    # Compensate and estimate signal power
    # FFT normalization in kbNUFFT MR operator is 2D by default. Multiplying
    # the energy term with nr adjusts it for 1D Radon projections.
    ksignal_ramp_mag = (magnitude(ksignal*kradius))
    ksignal_energy_coil = (
        ((ksignal_ramp_mag**2).sum((0, 2, 3)) * (1/(2*nspokes))) - pcomp
        ) * nr
    ksignal_power_coil = ksignal_energy_coil/(nr**2)
    ksignal_power = ksignal_power_coil.sum((0))

    # Calculate SNR (in dB)
    snr = 10*torch.log10(ksignal_power/knoise_power)

    return snr, ksignal_power, knoise_power
