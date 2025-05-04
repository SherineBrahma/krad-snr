import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
# matplotlib.use('Agg')


def plot_results(cfg) -> None:

    # Unpacking configuration
    nr = cfg["nr"]

    # Loading arrays
    save_path = str(Path(__file__).resolve().parent.parent.parent / "results")
    nspokes_array = np.load(Path.joinpath(Path(save_path), 'nspokes.npy'))
    SNR_est_array = np.load(Path.joinpath(Path(save_path), 'SNR_est.npy'))
    SNR_gnd_array = np.load(Path.joinpath(Path(save_path), 'SNR_gnd.npy'))
    
    # Color map
    cmap = matplotlib.cm.get_cmap('viridis_r')
    colors = cmap(np.linspace(0, 1, SNR_gnd_array.shape[0]))
    nspokes_max = int(np.ceil((np.pi/2)*nr))
    R_list = [int(i) for i in np.round(nspokes_max/nspokes_array)]
    
    # Plotting graph of true SNR vs estimated SNR
    fig = plt.figure(figsize=(5, 5))
    plt.title("Estimated SNR")
    for i, color in enumerate(colors):
        label = '$R$: ' + str(R_list[i])
        # if R_list[i] in [1,7]:
        #     continue
        if i == 0:
            plt.plot(SNR_gnd_array[i].reshape(-1),
                     SNR_gnd_array[i].reshape(-1),
                     linewidth=1,
                     linestyle="dashed",
                     label='Perfect estimation',
                     color='black')        
        plt.plot(SNR_gnd_array[i].reshape(-1),
                 np.nanmean(SNR_est_array[i], axis=-2).reshape(-1),
                 linewidth=1,
                 label=label,
                 color=color)
        SNR_est_mean = np.nanmean(SNR_est_array[i], axis=-2).reshape(-1)
        SNR_est_std = np.nanstd(SNR_est_array[i], axis=-2).reshape(-1)
        plt.fill_between(SNR_gnd_array[i].reshape(-1), 
                         SNR_est_mean - SNR_est_std, 
                         SNR_est_mean + SNR_est_std,
                         linewidth=1,
                         alpha=0.2,
                         color=color)
    plt.xlabel('$SNR_{\mathrm{GND}}$')
    plt.ylabel('$SNR_{\mathrm{EST}}$')
    plt.legend(loc="upper left")
    plt.grid()
    # plt.show()
    fig.savefig(Path.joinpath(Path(save_path), 'SNR_plot'), dpi=500)

    # Plotting graph of true SNR vs estimated SNR
    fig = plt.figure(figsize=(5, 5))
    plt.title("Estimated SNR")
    for i, color in enumerate(colors):
        label = '$R$: ' + str(R_list[i])
        # if R_list[i] in [1,7]:
        #     continue
        if i == 0:
            plt.plot(SNR_gnd_array[i].reshape(-1),
                     SNR_gnd_array[i].reshape(-1),
                     linewidth=1,
                     linestyle="dashed",
                     label='Perfect estimation',
                     color='black')        
        plt.plot(SNR_gnd_array[i].reshape(-1),
                 np.nanmean(SNR_est_array[i], axis=-2).reshape(-1),
                 linewidth=1,
                 label=label,
                 color=color)
        SNR_est_mean = np.nanmean(SNR_est_array[i], axis=-2).reshape(-1)
        SNR_est_std = np.nanstd(SNR_est_array[i], axis=-2).reshape(-1)
        plt.fill_between(SNR_gnd_array[i].reshape(-1), 
                         SNR_est_mean - SNR_est_std, 
                         SNR_est_mean + SNR_est_std,
                         linewidth=1,
                         alpha=0.2,
                         color=color)
    plt.xlabel('$SNR_{\mathrm{GND}}$')
    plt.ylabel('$SNR_{\mathrm{EST}}$')
    # plt.legend(loc="upper left")
    plt.ylim([5, 6.2])
    plt.xlim([5, 6.2])
    plt.yticks([5.5, 6])
    plt.xticks([5, 5.5, 6])
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    # plt.show()
    fig.savefig(Path.joinpath(Path(save_path), 'SNR_plot_zoomed'), dpi=500)
