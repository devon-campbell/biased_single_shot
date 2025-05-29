# Single-Shot Decoding of Biased-Tailored Quantum LDPC Codes

**Author**: Devon Campbell  
**Advisor**: [Henry Yuen](https://cs.columbia.edu/~hsy117/)  
**Thesis**: Undergraduate Thesis, Columbia University (Aug 2024 – May 2025)

---

## Overview

This repository implements a novel **four-dimensional quantum LDPC code** designed to tackle two central challenges in near-term quantum computing:

1. **Biased physical noise**, where certain error types dominate (e.g., dephasing).
2. **Faulty measurement**, where syndrome readout itself may be corrupted.

The construction leverages a **lifted homological product** of four classical protographs, embedding a redundant structure ("metachecks") that enables **single-shot decoding**—reliable error correction using a single round of noisy measurements.

---

## Features

- **Code Construction**: Builds 4D lifted homological-product quantum codes from protograph seeds.
- **Simulation Engine**: Evaluates decoding performance using circuit-level Monte Carlo methods under biased and noisy conditions.
- **Hadamard Tailoring**: Realigns stabilizers to the dominant error type using local basis changes.
- **Distance Estimation**: Uses Z3-optimized search to compute minimum logical weight.
- **Statistics + Sweeps**: Supports large-scale grid sweeps and postprocessing of decoding results.

---

## Repository Structure

```bash
.
├── mm_qc_pega/             # PEG-based protograph generator (input files and graph parser)
├── notebooks/
│ └── analysis.ipynb        # Jupyter notebook for visualizing results
├── plots/                  # Auto-generated plots from simulations
├── scripts/
│ ├── build_qcode.py        # Build and serialize a 4D lifted HGP code from QC protographs
│ ├── cached_pegs.py        # Construct and cache 4D matrices (H_X, H_Z, M_X, M_Z)
│ ├── compute_distance.py   # Compute code distance using Z3-based minimization
│ ├── css_ss_decode_sim.py  # Main simulation engine for single-shot CSS decoding
│ ├── distance.py           # Linear algebra utilities (RREF, generator matrix, etc.)
│ ├── lifted_hgp_4d.py      # Core definition of the 4D lifted homological product code
│ ├── sim_stats.py          # Simulation runner with performance + error stat tracking
│ └── sweep_grid_v1.py      # Parameter grid definition for full simulation sweeps
├── docs/
│   └── thesis\_report.pdf  # Full thesis writeup 
└── codes/                  # Output directory for generated codes and matrix cache
```

---

## Getting Started

### Requirements
- Python 3.8+
- Packages: `numpy`, `scipy`, `joblib`, `tqdm`, `z3-solver`, `pandas`, `matplotlib`, `bposd`, `ldpc`

You can install the core dependencies with:

```bash
pip install -r requirements.txt
````

### Build a Code

```bash
python build_qcode.py peg_6_4_3_0-1
```

This reads a `.qc` protograph file and stores the resulting quantum code under `codes/`.

### Run a Simulation

```bash
python sim_stats.py
```

Runs a batch simulation sweep over biased error channels and measurement error conditions. Results are appended to `results/results_v4.csv`.

---

## Results

* **Up to 60% reduction** in logical error under biased noise using Hadamard tailoring.
* **Single-shot decoding** recovers roughly one-third of the performance lost to measurement noise.
* Supports tuning tradeoffs between decoding success and false correction using metacheck statistics.

---

## Citation

If you find this code useful, please consider citing the corresponding thesis or GitHub repository:

> Devon Campbell, *Single-Shot Decoding of Biased-Tailored Quantum LDPC Codes*, Columbia University Undergraduate Thesis (2025). [GitHub](https://github.com/devon-campbell/biased_single_shot)

