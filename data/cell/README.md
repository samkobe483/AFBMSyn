# Cell Line Embedding with PPI Network

A specialized module for generating cell line embeddings by integrating Protein-Protein Interaction (PPI) networks with multi-omics data (gene expression and mutation) to support drug synergy prediction.

## Overview

This sub-project implements the embedding logic based on the PRODeepSyn framework:
- **GINEncoder**: Utilizes Graph Isomorphism Networks to capture the topological structure of PPI graphs.
- **Cell2Vec**: A hybrid model that merges PPI network features with learnable cell line embeddings.
- **Multi-Omics Integration**: Supports both Gene Expression (GE) and Mutation (MUT) data pathways.

## File Description

| File | Description |
| :--- | :--- |
| `const.py` | Configuration for data paths and constants. |
| `dataset.py` | Dataset classes including `C2VDataset` and `C2VSymDataset`. |
| `model.py` | Core architectures: `GINEncoder`, `Cell2Vec`, and `RandomW`. |
| `train.py` | Main training script for generating cell embeddings. |
| `gen_feat.py` | Script to generate and normalize cell features from trained models. |
| `utils.py` | Utility functions for model persistence and loss visualization. |
| `train_gin_example.py` | Example script demonstrating GIN encoder implementation. |

## Installation & Data Preparation

### 1. Requirements
Ensure your environment includes `torch` and `torch_geometric` (refer to the main `environment.yaml`).

### 2. Data Setup
Place the following files in `data/Cell/data/data/`:
- `ppi.coo.npy`: PPI network edges in COO format.
- `node_features.npy`: Initial node features for the PPI graph.
- `target_ge.npy` / `nodes_ge.npy`: Gene expression targets and corresponding nodes.
- `target_mut.npy` / `nodes_mut.npy`: Mutation targets and corresponding nodes.

## Usage

### Step 1: Train Cell Embeddings
Train the models for both Gene Expression (GE) and Mutation (MUT) pathways.
```bash
# Default config: 128 hidden dim, 384 embedding dim
python train.py
