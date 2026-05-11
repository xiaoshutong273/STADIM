# STADIM

**S**patial **T**ranscriptomics **A**daptive **D**enoising and **I**ntegration for **M**ulti-slice data

STADIM is a deep learning framework for transcriptome-wide consistent denoising and integration of spatial transcriptomics data.

<p align="center">
  <img width="3909" height="2873" alt="Fig1" src="https://github.com/user-attachments/assets/3563c116-1070-426c-bccc-0ebdcab0bd22" />
</p>

## Overview

Spatial transcriptomics (ST) enables high-resolution mapping of gene expression across tissues, yet its application is limited by pervasive technical noise, batch effects, and the lack of unified models for joint denoising and integration. Here, we present STADIM, a deep-learning framework for consistent representation of multi-slice ST data through transcriptome-wide denoising and integration. STADIM operates directly in high-dimensional gene expression space and employs a dual-branch architecture to explicitly disentangle biological signals from technical variation. By integrating latent biological embeddings with batch-specific representations and reconstructing expression profiles under a negative binomial model beyond highly variable genes, STADIM effectively reduces stochastic noise while preserving fine-grained spatial structure. In addition, an adaptive triplet learning strategy enhances cross-slice alignment and robustness across heterogeneous datasets. Extensive evaluations across diverse tissues, disease states, and sequencing platforms demonstrate that STADIM consistently outperforms state-of-the-art methods in denoising, batch correction, and spatial structure preservation, while maintaining biologically meaningful patterns such as tumor microenvironment heterogeneity, tertiary lymphoid structures, and cardiovascular inflammatory pathways.

## Installation

```bash
# Create a new environment
conda create -n stadim python=3.9 -y
conda activate stadim

## Install STADIM directly from GitHub
pip install git+https://github.com/xiaoshutong273/STADIM.git

## or download the zip from GitHub and unzip the folder
cd STADIM-main
pip install .

# To use the environment in jupyter notebook, add python kernel for this environment.
pip install ipykernel
python -m ipykernel install --user --name=STADIM
```

The code has been successfully tested on macOS and Ubuntu 24.04. We highly recommend running the program on a GPU-enabled device for optimal performance.

## Quick Start

Once installed, you can call STADIM directly from your terminal:

```bash
stadim --input ./data.h5ad --save_preprocessed_h5ad ./data_preprocessed.h5ad --save_dir ./output --batch_key "sample" device "cuda:0" --monitor --seed 2026     
```

## Detailed Parameter Guide

### 1. Data Input/Output
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | **Required** | Path(s) to input `.h5ad` file(s). Supports single file like `"./data.h5ad"`, or multiple paths like `"./data1.h5ad" "./data2.h5ad" "./data3.h5ad"`. |
| `--save_preprocessed_h5ad` | `str` | None | Filename (ending in `.h5ad`) to save the data after initial filtering and preprocessing. This file serves as the clean input for the training model. If you do not wish to save the filtered intermediate file to disk, omit it. |
| `--save_dir` | `str` | None | Directory where all output results will be stored. |
| `--batch_key` | `str` | `"sample"` | The column name in `adata.obs` used to distinguish different batches or samples. If not, you can set it to `None` and add that information in `--sample_names`. |
| `--sample_names` | `str` | `None` | Used to name and distinguish batches if no sample column exists in `adata.obs`. If omitted, samples are named S1, S2, S3... by input order and stored under the `"sample"` key. |

### 2. Preprocessing Filters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--min_genes` | `int` | `0` | Filter out cells with fewer than this many genes. |
| `--min_cells` | `int` | `10` | Filter out genes expressed in fewer than this many cells. |
| `--nhvgs` | `int` | `2000` | Number of Highly Variable Genes to select for PCA analysis. |
| `--dim` | `int` | `50` | Number of Principal Components (PCs) used for initial neighbor searching. |

### 3. Triplet Sampling
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--knn_c` | `int` | `6` | **KNN (Spatial)**. The number of nearest neighbors identified based on spatial coordinates. |
| `--knn_e` | `int` | `10` | **KNN (Expression)**. The number of nearest neighbors identified based on gene expression. |
| `--mnn_n` | `int` | `25` | **MNN (Expression)**. Number of Mutual Nearest Neighbors used to align different batches. |
| `--hard_triplets_ratio` | `float` | `0.7` | The target percentage of "hard" triplets in the dataset. |
| `--triplets_update_ratio`| `float` | `0.8` | How frequently the triplet updates during training. |

### 4. Model Architecture
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--encoder_layers` | `int list`| `512 256 64` | Hidden layer dimensions for the Encoder. |
| `--decoder_layers` | `int list`| `1000` | Hidden layer dimensions for the Decoder. |
| `--loss_mode` | `str` | `"nb"` | Distribution model for denoising. Choose `nb` (Negative Binomial) or `zinb` (Zero-Inflated NB). |
| `--beta_trip` | `float` | `0.1` | Weight of the Triplet Loss relative to the Reconstruction Loss. |
| `--epochs` | `int` | `100` | Total number of training iterations. |
| `--lr` | `float` | `1e-3` | Learning rate for the Adam optimizer. |
| `--batch_size` | `int` | `256` | Number of training samples per optimization step. |

### 5. Monitoring
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--device` | `str` | `"cuda:0"` | Device to use. Supports `cpu` or `cuda:X`. |
| `--monitor` | `flag` | `False` | If set, records CPU/GPU/Memory usage in a background thread during execution. |
| `--seed` | `int` | `2026` | Random seed for reproducibility. |

## Results Guide

`--save_dir` output folder contains the main output file as follows:

| File | Description |
| :--- | :--- |
| **`sdata.h5ad`** | Main Output. Integrated AnnData object containing denoised counts in `layers['STADIM']` and latent embeddings in `obsm['STADIM']`. |
| **`STADIM_denoised.npy`** | Raw NumPy matrix of the full-transcriptome denoised expression data `sdata.layers['STADIM']`. |
| **`STADIM_embed.npy`** | Raw NumPy matrix of the low-dimensional latent representations `sdata.obsm['STADIM']`. |
| **`trained_model.pth`** | The trained PyTorch model for future inference or fine-tuning. |
| **`training_loss.png`** | Visualization of the loss curve across epochs. |
| **`other_variables.npz`** | Compressed archive of training hyperparameters and intermediate process variables. |
| **`resource_trace.txt`** | Time-series log of CPU, GPU, and RAM usage (generated if `--monitor` is enabled). |
| **`performance.csv`** | Summary of peak hardware consumption metrics (generated if `--monitor` is enabled). |

## Tutorials

To replicate the results presented in our paper or to learn how to use STADIM step-by-step, please refer to: https://stadim.readthedocs.io/en/latest/index.html

You can also test the full pipeline using the provided demo dataset (Mouse Olfactory Bulb from Stereo-seq and Slide-seq V2 platforms):

```bash
nohup stadim --input "./MOB_stereoseq.h5ad" "./MOB_slideseq.h5ad" \
    --save_preprocessed_h5ad ./MOB_test/data_preprocessed.h5ad --save_dir ./MOB_test --device "cuda:0" --monitor --seed 2026 \
    --min_genes 10 --min_cells 10 --knn_c 30 --knn_e 70 --mnn_n 100 > ./MOB_test.log 2>&1 &
```
