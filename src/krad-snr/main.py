from pathlib import Path
from typing import Dict

import yaml
from plot_results import plot_results
from simulation_manager import run_simulation_batch


def load_cfg() -> Dict:
    """
    Load the simulation configuration from a YAML file.

    Returns:
        cfg (Dict): Configuration parameters loaded from 'config/config.yaml'.
    """

    # Read config file
    cfg_path = str(
        Path(__file__).resolve().parent.parent.parent / "config/config.yaml"
        )
    with open(cfg_path) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return cfg


def main() -> None:
    """
    Main execution function:
        1. Load configuration.
        2. Run a batch of simulations.
        3. Plot the simulation results.
    """

    # Load configuration
    cfg = load_cfg()

    # Run simulation for varying noise levels
    # and number of spokes (acceleration factors)
    run_simulation_batch(cfg)

    # Plot graphs of the true SNR vs estimated SNR
    plot_results(cfg)


if __name__ == "__main__":
    main()
