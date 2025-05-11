# SNR in Radial Multi-Coil MRI
This repository provides a method for estimating SNR directly from k-space data in radial multi-coil MRI. Unlike conventional image-domain methods, it is independent of reconstruction algorithms and unaffected by noise inhomogeneity across the MRI image introduced by multicoil reconstruction. It is less sensitive to undersampling factors, providing more reliable SNR estimation, which is useful for tasks such as generating realistic noise levels in simulated k-space datasets and evaluating pulse sequences, RF coils, and MR systems across different field strengths. A brief overview is provided in the video below.

https://github.com/user-attachments/assets/829885ef-0619-4d6d-bbd7-74af3e20fd57

<a id="publication"></a>
| [Publication](https://ieeexplore.ieee.org/document/10731565) | [Citation](#bibtex-citation) |

**Contribution**: Sherine Brahma, Christoph Kolbitsch, and Tobias Schaeffter.

# Installation

### 1. Clone the repository
Clone the repository and create a new Python environment with Python 3.8 (e.g. using conda):
```bash
git clone https://github.com/SherineBrahma/krad-snr.git
conda create -n krad-snr python=3.8
conda activate krad-snr
```

### 2. Install KRadSNR and dependencies
Install KRadSNR in editable mode along with necessary tools for linting, testing. 
```bash 
pip install -e ".[lint,test]"
```

# Usage

## 1. Simulation

The simulation demonstrates the SNR estimation method using a Shepp-Logan phantom with 320×320 dimensions. Radial k-space spokes are generated using the NUFFT operator from the TorchKbNufft package, applying a golden-angle acquisition scheme with 32 dummy coil sensitivity maps and varying undersampling factors. Gaussian noise with different standard deviations is then added to the k-space data to evaluate the proposed method under various simulated SNR levels. Configuration inputs for the simulation are defined in the config files located in the config folder. An explanation of these inputs is provided below:

```yaml 
# General Parameters
nr: 320                # Number of readout points per frame
nt: 3                  # Number of temporal frames
osmpl_fact: 2          # Oversampling factor
nspokes_array: [503, 251, 75, 50, 36]  # List of total radial spokes to be simulated
noise_std_list: [1.0, 0.8, 0.5, 0.2, 0.1, 0.05]  # Noise standard deviations
nbootstrap: 100        # Number of bootstrap repetitions for SNR estimation
device: 'cuda'         # Computation device ('cuda' or 'cpu')
```

To run the simulation, execute the following command:

```python
python src/krad-snr/main.py
```

After running the experiment a folder named ```results``` is created and the simulation results are stored here. An example of a plot is shown below on the right for the shepp logan phantom on the left.

<div align="center">
  <img src="media/snr_plot.png" width="700" height="auto">
</div>

## 2. Custom Data

A pre-trained model, trained on a larger dataset (see [publication](#publication) for details), is included for quick testing. You can directly run the provided testing script to evaluate the pre-trained model on two datasets: i) with motion artifacts, and ii) without motion artifacts.

```python
sh script/test_job_queue.sh
```

The results will be saved as output arrays in the ```experiments``` folder, which can be further assessed. For example, to visualize the generated arrays, run:

```python
 python src/deepfermi/analysis/generate_img.py
```

The example below shows that DeepFermi estimates are more robust to motion artifacts compared to traditional Fermi-deconvolution, which relies on well-established optimization algorithms without deep learning components, such as the [Limited memory Broyden-Fletcher-Goldfarb-Shanno](https://link.springer.com/article/10.1007/BF01589116) (LBFGS) algorithm.

<div align="center">
  <img src="media/results.png" width="700" height="auto">
</div>  

You can also write custom scripts to analyze the arrays. Additionally, an ```evaluate_measures.py script``` is provided for quantitatively assessing the performance of the model.

# Citation

<a id="bibtex-citation"></a>

If you found our work useful, please consider citing it. The BibTeX is provided below:

```bibtex
@article{brahma2024robust,
  title={Robust Myocardial Perfusion MRI Quantification with DeepFermi},
  author={Brahma, Sherine and Kolbitsch, Christoph and Schaeffter, Tobias},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2025},
  publisher={IEEE}
}
```
