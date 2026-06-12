# TAP-CDP: Task-Aware Compression for Compressed-Domain Processing

This repository contains the official PyTorch implementation of the **TAP-CDP** framework, as described in our research. TAP-CDP is a task-aware spectral sparsification framework designed for efficient inference directly on compressed-domain (DCT) signals.

## Experimental Setup

Our evaluation protocol ensures a highly rigorous, fair, and reproducible comparison across standard vision benchmarks.

### Key Features
- **Backbones**: Modified ResNet-18 and ResNet-50 for DCT-domain inference.
- **TACS Module**: Learned spectral channel selection with linear temperature annealing and Straight-Through Estimation (STE).
- **Optimization**: SGD with Nesterov momentum, Cosine Annealing scheduler, and Mixed Precision (FP16) support.
- **Hardware Metrics**: Simulated inference latency and energy consumption for NVIDIA T4 GPU.

## Repository Structure

- `tap_hybridnet.py`: Core architecture including the TACS module and Modified ResNet backbones.
- `train.py`: Training protocol implementation with support for mixed precision and hardware metrics.
- `main.py`: Driver script for running ablation studies across multiple independent seeds.
- `dataset.py`: Data loading and preprocessing for CIFAR-10 and ImageNet.
- `config.yaml`: Centralized experiment configuration.
- `compression.py`: Learned compression baseline components.
- `assets/`: Research figures, architectural diagrams, and visualization results.
- `scripts/`: Utility scripts for additional analysis and visualization.
- `results/`: Output directory for logs, models, and evaluation plots (ignored by git).

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Dataset Preparation
Prepare the data directories and download the datasets using the provided script:
```bash
./datasets_setup.sh
```
*Note: CIFAR-10 will be downloaded automatically during the first run. For ImageNet, please follow the manual download instructions printed by the script.*

### Running Experiments
To replicate the full ablation study across 3 seeds (as defined in `config.yaml`):
```bash
python3 main.py
```

## Results
The framework is evaluated across the **Rate–Accuracy–Complexity (R–A–C) Pareto space**:
- **Task Accuracy**: Top-1 Accuracy.
- **Bit-Rate (BPP)**: Entropy-based latent representation cost.
- **Sparsity Ratio**: Fraction of active spectral components.
- **Efficiency**: GFLOPs, Latency (ms), and Energy (mJ).

## Reproducibility
All experiments are seeded deterministically (default seed: 42). Results reflect the mean and standard deviation across three independent runs.
