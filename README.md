# Weighted Potential CRPS

Code accompanying the paper **“Towards Fair Comparisons of AI- and Physics-Based Weather Models for Extreme Events via the Weighted Potential CRPS”**.

The repository contains code to evaluate deterministic weather forecasts with the potential CRPS (PCRPS) and weighted extensions of the PCRPS. The main application is the comparison of AI weather prediction models and physics-based numerical weather prediction models for extreme weather events.

## Overview

The main scores implemented here are:

* `PCRPS`: potential CRPS.
* `twPCRPS`: threshold-weighted PCRPS.
* `qwPCRPS`: quantile-weighted PCRPS.
* `PCRPS0`, `twPCRPS0`, and `qwPCRPS0`: unconditional climatology reference scores used to compute skill scores.

## Repository structure

```text
pcrps/
  compute_pcrps.py       Compute PCRPS, twPCRPS, and qwPCRPS
  compute_pcrps0.py      Compute climatological reference scores
  pcrps_helper.py        Score calculation utilities
  test_scores.py         Block permutation tests

scripts/
  compute_thresholds.py  Build ERA5 quantile and record thresholds
  evaluate_ensemble.py   Evaluate ensemble forecasts

jobs/
  run_*.sh               Shell scripts used for the paper experiments
```

## Setup

The code expects local paths to data, thresholds, and output directories. Create a local paths file from the example:

```bash
cp local_paths.env.example local_paths.env
```

Then edit `local_paths.env` and set the paths for your system.

Install the required Python environment with:

```bash
conda env create -f environment.yml
conda activate weighted_pcrps
```

## Usage

The shell scripts in `jobs/` contain the commands used for the experiments in the paper. They automatically change to the repository root and load `local_paths.env`.

For example, to compute PCRPS-based scores:

```bash
bash jobs/run_pcrps.sh
```

To compute reference scores:

```bash
bash jobs/run_pcrps0.sh
```

To build threshold datasets directly from Python, first load the local path configuration:

```bash
source local_paths.env
python scripts/compute_thresholds.py
```

## Data

The experiments use WeatherBench 2 forecast and observation data together with ERA5-derived threshold datasets. The scripts are written for Zarr datasets and support both local paths and cloud-hosted datasets where applicable.

## Paper

**Biegert et al. (2026).** Towards Fair Comparisons of AI- and Physics-Based Weather Models for Extreme Events via the Weighted Potential CRPS.<br>
Preprint available on [arXiv:2606.21170](https://arxiv.org/abs/2606.21170).

### Citation

If you use this code, please cite the accompanying paper:

```bibtex
@misc{biegert2026faircomparisonsaiphysicsbased,
  title={Towards Fair Comparisons of AI- and Physics-Based Weather Models for Extreme Events via the Weighted Potential CRPS},
  author={Tobias Biegert and Sam Allen and Annika Alber and Sebastian Lerch},
  year={2026},
  eprint={2606.21170},
  archivePrefix={arXiv},
  primaryClass={stat.AP},
  url={https://arxiv.org/abs/2606.21170},
}
```
