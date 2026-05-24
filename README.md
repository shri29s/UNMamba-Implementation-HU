# UNMamba: Cascaded Spatial-Spectral Mamba for Blind Hyperspectral Unmixing

This repository contains a Python baseline implementation of UNMamba, a cascaded spatial-spectral Mamba architecture for blind hyperspectral unmixing (HU), built with the mambapy library.

> Based on the paper:  
> UNMamba: Cascaded Spatial-Spectral Mamba for Blind Hyperspectral Unmixing  
> IEEE Geoscience and Remote Sensing Letters, 2025  
> IEEE Xplore: https://ieeexplore.ieee.org/document/10902420

## Overview

Blind hyperspectral unmixing decomposes a hyperspectral image into endmember spectra and abundance maps without requiring prior endmember initialization.

UNMamba addresses limitations of CNN (localized receptive field) and Transformer (quadratic complexity) based unmixing methods by leveraging **State Space Models (SSMs)** via the Mamba architecture, which captures long-range spatial-spectral dependencies with **linear computational complexity**.

### Key Features

- **Cascaded Spatial-Spectral Mamba blocks** — spatial dependencies are captured first, followed by spectral feature extraction
- **No endmember initialization** — does not rely on VCA or any external initialization technique
- **Linear Mixing Model (LMM)** reconstruction with trainable random spectral sequences
- **Endmember loss** for learning discriminative endmember spectra
- **Efficient and lightweight** — small parameter count and low FLOPs compared to Transformer-based methods
- **First SSM-based unmixing method** in the hyperspectral literature

---

## Architecture

```
Input HSI (H × W × B)
        │
        ▼
┌───────────────────┐
│  Spatial Mamba    │  ← Long-range spatial dependency modeling
│     Blocks        │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Spectral Mamba   │  ← Global spectral feature extraction
│     Blocks        │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Abundance Head   │  ← Maps features → abundance maps (H × W × P)
└───────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  LMM Reconstruction         │  ← Weighted avg of trainable sequences
│  Ŷ = A · E  +  endmember    │    + endmember loss
│         loss                │
└─────────────────────────────┘
```

Where **P** = number of endmembers, **B** = number of spectral bands.

---

## Requirements

- Python 3.8 or newer
- torch
- mambapy
- numpy
- scipy
- matplotlib
- ipykernel

Install the dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** `mambapy` is a pure-Python/PyTorch Mamba implementation that does not require CUDA-specific kernels, making it portable across environments.

---

## Installation

```bash
git clone https://github.com/shri29s/UNMamba-Implementation-HU.git
cd UNMamba-Implementation-HU
pip install -r requirements.txt
```

## Dataset Format

The training script expects the following dataset files under data/:

| Dataset     | File                 | HSI key | Spatial size | Bands | Endmembers |
| ----------- | -------------------- | ------- | ------------ | ----- | ---------- |
| Jasper      | data/jasper.mat      | Y       | 100 x 100    | 198   | 4          |
| Cuprite     | data/cuprite.mat     | X       | 512 x 614    | 188   | 4          |
| Chandrayaan | data/chandrayaan.mat | Y       | 2995 x 304   | 85    | 4          |

If present, the .mat files may also include ground-truth abundance maps under A and endmembers under M. The loader reads those fields automatically when available.

## Usage

Run training from the repository root:

```bash
python src/train.py --dataset jasper --epochs 800 --lr 3e-3 --batch_size 16 --patch_size 64 --stride 32
```

Supported dataset values are jasper, cuprite, and chandrayaan.

Useful arguments:

| Argument     | Default | Description                                           |
| ------------ | ------- | ----------------------------------------------------- |
| --dataset    | jasper  | Dataset to train on                                   |
| --epochs     | 800     | Number of epochs                                      |
| --lr         | 3e-3    | Learning rate                                         |
| --batch_size | 16      | Batch size used when patching is enabled              |
| --patch_size | 64      | Patch size for training; set to 0 to disable patching |
| --stride     | 32      | Patch stride                                          |

## Outputs

Each run saves a compressed result file in the repository root:

- unmixing_results_jasper.npz
- unmixing_results_cuprite.npz
- unmixing_results_chandrayaan.npz

Plots are written under results/ using the dataset name chosen by the loader:

- results/jasper/
- results/Cuprite/
- results/Chandrayaan/

These folders contain abundance maps, estimated endmember spectra, and the training loss curve.

## Metrics

The training script reports:

- Reconstruction RMSE
- SAD for endmember quality when ground truth endmembers are available
- SID for endmember comparison when ground truth endmembers are available
- Abundance RMSE when ground truth abundance maps are available

## Results From Saved Runs

The repository includes saved outputs in the `unmixing_results_*.npz` files. The table below summarizes the metrics stored in those archives.

| Dataset     | Reconstruction RMSE | Mean SAD (deg) | Mean SID | Mean Abundance RMSE |
| ----------- | ------------------: | -------------: | -------: | ------------------: |
| Jasper      |            0.046587 |       3.584849 | 0.008246 |            0.067916 |
| Cuprite     |            0.018255 |      10.789442 | 0.056497 |                 N/A |
| Chandrayaan |            0.033461 |            N/A |      N/A |                 N/A |

## Project Structure

```
UNMamba/
├── data/
├── notebooks/
│   ├── explore_dataset.ipynb
│   └── explore_results.ipynb
├── results/
├── src/
│   ├── loader.py
│   ├── loss.py
│   ├── metrics.py
│   ├── train.py
│   ├── unmamba.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
D. Chen, J. Zhang and J. Li, "UNMamba: Cascaded Spatial–Spectral Mamba for Blind Hyperspectral Unmixing,"
in IEEE Geoscience and Remote Sensing Letters, vol. 22, pp. 1-5, 2025, Art no. 5502405,
doi: 10.1109/LGRS.2025.3545505.
keywords: {Computational modeling;Feature extraction;Estimation;Transformers;
Hyperspectral imaging; Training;Random sequences;Geoscience and remote sensing;
Decoding;Data mining;Blind hyperspectral unmixing (HU);
endmember loss;linear mixing model (LMM);Mamba;state-space model},
```

## Acknowledgements

- Original paper authors for the UNMamba architecture
- [`mambapy`](https://github.com/alxndrTL/mamba.py) for the pure-Python Mamba implementation
- The hyperspectral unmixing community for benchmark datasets
