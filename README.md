# DSPSCL: Drug Synergy Prediction via Self-Supervised Contrastive Learning

A deep learning framework for anti-cancer drug combination synergy prediction using multi-source information fusion and attention mechanisms.

## Overview

DSPSCL integrates:
- **Drug features**: Chemical structure (Morgan fingerprints), target proteins, and pathway information
- **Cell line features**: Gene expression and mutation data via autoencoder embeddings
- **Attention mechanism**: Multi-head self-attention for learning combined drug representations
- **Contrastive learning**: Self-supervised pre-training for better feature representations

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/DSPSCL.git
cd DSPSCL
```

### 2. Create conda environment

```bash
conda env create -f environment.yaml
```

### 3. Activate environment

```bash
conda activate DSPSCL
```

## Dataset

The project uses drug synergy data from:
- **Oneil dataset**: Drug combinations from O'Neil et al.
- **ALMANAC dataset**: NCI-ALMANAC drug synergy data

Place your data files in the `data/` directory:
- `smiles.csv` - Drug SMILES structures
- `drug_protein_feature.pkl` - Drug-target protein features
- `drug_pathway_feature.pkl` - Drug-pathway features
- `cell_features.csv` - Cell line gene expression data
- `cell_feat.npy` - Cell line feature matrix
- `oneil_synergyloewe.txt` / `almanac_synergyloewe.txt` - Synergy scores

## Usage

### Training

```bash
# Train the model with 5-fold cross-validation
python main.py
```

### Evaluation

```bash
# Run ablation study
python ablation_study.py

# Cell line analysis
python patch_cell_line_analysis.py

# Visualization
python keshihua.py
```

### Configuration

Modify `get_dataset.py` to switch datasets:
```python
SYNERGY_FILENAME = 'oneil_synergyloewe.txt'  # or 'almanac_synergyloewe.txt'
```

## Project Structure

```
DSPSCL/
├── data/                    # Data files
├── results/                 # Training results
├── attention_vis/           # Attention visualization
├── main.py                  # Main training script
├── model.py                 # DSPSCL model
├── get_dataset.py           # Data loading and processing
├── ablation_study.py        # Ablation experiments
├── keshihua.py              # Visualization
└── environment.yaml         # Conda environment
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- RDKit
- PyTorch Geometric
- See `environment.yaml` for full dependencies

## Citation

If you use this code, please cite our work.
