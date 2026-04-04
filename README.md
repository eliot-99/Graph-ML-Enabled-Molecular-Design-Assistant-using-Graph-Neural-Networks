---
title: MolGNN Tox21 Predictor
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# 🧬 Graph ML-Enabled Molecular Design Assistant using Graph Neural Networks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-5C5C5C?style=for-the-badge)](https://www.rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**🔬 An AI-powered molecular toxicity prediction web application using state-of-the-art Graph Neural Networks**

[🚀 Live Demo](https://huggingface.co/spaces/Epion09g/MolGNN-Tox21-Predictor) • [📖 Documentation](#-usage) • [🧠 Models](#-model-architectures) • [📊 Results](#-model-performance)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Live Demo](#-live-demo)
- [Tox21 Endpoints](#-tox21-endpoints)
- [Model Architectures](#-model-architectures)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Deployment](#-deployment)
- [Team](#-team)
- [References](#-references)
- [License](#-license)

---

## 🎯 Overview

**MolGNN Tox21 Predictor** is a comprehensive molecular toxicity prediction system that leverages Graph Neural Networks (GNNs) to assess the potential toxicity of chemical compounds. The system predicts toxicity across all **12 endpoints** of the **Tox21 benchmark dataset**, which is widely used in drug discovery, pharmaceutical research, and chemical safety assessment.

### Why Graph Neural Networks for Molecules?

Molecules are naturally represented as graphs where:
- **Nodes** = Atoms (with features like atomic number, charge, hybridization)
- **Edges** = Chemical bonds (with features like bond type, aromaticity)

GNNs can learn molecular representations directly from this graph structure, capturing both local atomic environments and global molecular properties without requiring hand-crafted molecular descriptors.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Multi-Model Ensemble** | Choose from 3 GNN architectures (GINE, GCN, GATv2) or use all models together |
| 🎨 **Interactive 3D Visualization** | Rotate, zoom, and explore molecules in 3D using 3Dmol.js |
| 📊 **12 Toxicity Endpoints** | Comprehensive prediction across all Tox21 assays |
| 📁 **Batch Processing** | Upload CSV files with SMILES for bulk predictions |
| 💾 **Export Results** | Download predictions as CSV for further analysis |
| 🌐 **Modern Web UI** | Beautiful dark-themed responsive interface with Bootstrap 5 |
| ⚡ **Fast Inference** | Optimized PyTorch models for quick predictions |

---

## 🚀 Live Demo

**Try the application now:** [https://huggingface.co/spaces/Epion09g/MolGNN-Tox21-Predictor](https://huggingface.co/spaces/Epion09g/MolGNN-Tox21-Predictor)

### Quick Start Examples

Try these SMILES strings in the demo:

| Molecule | SMILES | Description |
|----------|--------|-------------|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | Common pain reliever |
| Caffeine | `Cn1cnc2c1c(=O)n(c(=O)n2C)C` | Stimulant in coffee |
| Ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` | Anti-inflammatory drug |
| Ethanol | `CCO` | Alcohol |
| Benzene | `c1ccccc1` | Aromatic hydrocarbon |
| Paracetamol | `CC(=O)Nc1ccc(O)cc1` | Acetaminophen |

---

## 🎯 Tox21 Endpoints

The Tox21 dataset contains toxicity labels for 12 biological assays, divided into two categories:

### Nuclear Receptor (NR) Panel

| Endpoint | Full Name | Biological Significance |
|----------|-----------|------------------------|
| **NR-AR** | Androgen Receptor | Male hormone signaling, endocrine disruption |
| **NR-AR-LBD** | AR Ligand Binding Domain | Direct AR binding activity |
| **NR-AhR** | Aryl Hydrocarbon Receptor | Xenobiotic metabolism, dioxin-like toxicity |
| **NR-Aromatase** | Aromatase Enzyme | Estrogen biosynthesis inhibition |
| **NR-ER** | Estrogen Receptor | Female hormone signaling, endocrine disruption |
| **NR-ER-LBD** | ER Ligand Binding Domain | Direct ER binding activity |
| **NR-PPAR-gamma** | PPAR-gamma | Lipid metabolism, diabetes-related |

### Stress Response (SR) Panel

| Endpoint | Full Name | Biological Significance |
|----------|-----------|------------------------|
| **SR-ARE** | Antioxidant Response Element | Oxidative stress response |
| **SR-ATAD5** | ATAD5 | Genotoxicity, DNA damage response |
| **SR-HSE** | Heat Shock Element | Cellular stress response |
| **SR-MMP** | Mitochondrial Membrane Potential | Mitochondrial toxicity |
| **SR-p53** | p53 Tumor Suppressor | Genotoxicity, apoptosis activation |

---

## 🧠 Model Architectures

We implemented and compared three state-of-the-art GNN architectures:

### 1. GINE (Graph Isomorphism Network with Edge features)

```
Input → GINEConv(256) → BatchNorm → ReLU → Dropout
      → GINEConv(256) → BatchNorm → ReLU → Dropout
      → GINEConv(256) → BatchNorm → ReLU → Dropout
      → GINEConv(256) → BatchNorm → ReLU → Dropout
      → [MeanPool || SumPool] → Linear(512→256) → Linear(256→12)
```

**Key Features:**
- Incorporates edge attributes (bond types)
- Based on Weisfeiler-Lehman graph isomorphism test
- Best theoretical expressiveness among the three

### 2. GCN (Graph Convolutional Network)

```
Input → GCNConv(256) → BatchNorm → ReLU → Dropout
      → GCNConv(256) → BatchNorm → ReLU → Dropout
      → GCNConv(256) → BatchNorm → ReLU → Dropout
      → GCNConv(256) → BatchNorm → ReLU → Dropout
      → [MeanPool || SumPool] → Linear(512→256) → Linear(256→12)
```

**Key Features:**
- Classical spectral-based convolution
- Efficient and well-understood
- Good baseline performance

### 3. GATv2 (Graph Attention Network v2)

```
Input → GATv2Conv(256, heads=4) → BatchNorm → ELU → Dropout
      → GATv2Conv(256, heads=4) → BatchNorm → ELU → Dropout
      → GATv2Conv(256, heads=4) → BatchNorm → ELU → Dropout
      → GATv2Conv(256, heads=4) → BatchNorm → ELU → Dropout
      → [MeanPool || SumPool] → Linear(512→256) → Linear(256→12)
```

**Key Features:**
- Dynamic attention mechanism
- Multi-head attention (4 heads)
- Learns to weight neighbor contributions

---

## 📊 Model Performance

### Scaffold Split Results (Recommended for Realistic Evaluation)

| Model | Macro AUC-ROC | Best Endpoints | Training Time |
|-------|---------------|----------------|---------------|
| **GINE** | **0.78** | SR-MMP (0.89), NR-AhR (0.86) | ~45 min |
| GATv2 | 0.76 | SR-p53 (0.84), SR-ARE (0.82) | ~60 min |
| GCN | 0.75 | NR-Aromatase (0.81), SR-HSE (0.80) | ~30 min |

### Per-Endpoint Performance (GINE Model)

| Endpoint | AUC-ROC | Precision | Recall |
|----------|---------|-----------|--------|
| NR-AR | 0.74 | 0.68 | 0.71 |
| NR-AR-LBD | 0.82 | 0.75 | 0.78 |
| NR-AhR | 0.86 | 0.79 | 0.82 |
| NR-Aromatase | 0.77 | 0.70 | 0.74 |
| NR-ER | 0.73 | 0.66 | 0.70 |
| NR-ER-LBD | 0.79 | 0.72 | 0.76 |
| NR-PPAR-gamma | 0.71 | 0.64 | 0.68 |
| SR-ARE | 0.80 | 0.73 | 0.77 |
| SR-ATAD5 | 0.75 | 0.68 | 0.72 |
| SR-HSE | 0.78 | 0.71 | 0.75 |
| SR-MMP | 0.89 | 0.83 | 0.86 |
| SR-p53 | 0.82 | 0.75 | 0.79 |

---

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/eliot-99/Graph-ML-Enabled-Molecular-Design-Assistant-using-Graph-Neural-Networks.git
cd Graph-ML-Enabled-Molecular-Design-Assistant-using-Graph-Neural-Networks
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install PyTorch Geometric (if needed)

```bash
# For CPU
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# For CUDA 11.8
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Step 5: Run the Application

```bash
python app.py
```

Open your browser and navigate to: `http://localhost:7860`

---

## 📖 Usage

### Single Molecule Prediction

1. Enter a **SMILES string** in the input field
2. Select a **model** (GINE, GCN, GATv2, or All)
3. Click **"Predict"**
4. View results with 2D/3D molecular visualization

### Batch Prediction

1. Prepare a CSV file with a column named `SMILES`:
   ```csv
   SMILES
   CCO
   CC(=O)Oc1ccccc1C(=O)O
   Cn1cnc2c1c(=O)n(c(=O)n2C)C
   ```
2. Upload the CSV file
3. Select model(s) for prediction
4. Download results as CSV

### API Usage (for Developers)

```python
import requests

# Single prediction
response = requests.post('http://localhost:7860/predict', data={
    'smiles': 'CCO',
    'model': 'gnn'  # Options: gnn, gcn, gatv2, all
})
results = response.json()
```

---

## 📁 Project Structure

```
Graph-ML-Enabled-Molecular-Design-Assistant/
│
├── 📄 app.py                    # Flask web application
├── 📄 requirements.txt          # Python dependencies
├── 📄 Dockerfile               # Docker configuration
├── 📄 README.md                # This file
├── 📄 LICENSE                  # MIT License
│
├── 📁 models/                  # Pre-trained model weights
│   ├── best_gnn_tox21_scaffold.pt    # GINE model (~2MB)
│   ├── best_gcn_tox21_scaffold.pt    # GCN model (~2MB)
│   └── best_gatv2_tox21_scaffold.pt  # GATv2 model (~2MB)
│
├── 📁 templates/               # HTML templates
│   ├── index.html             # Main prediction interface
│   └── about.html             # About page
│
├── 📁 notebooks/               # Training notebooks
│   ├── GAT_Tox21.ipynb
│   ├── Tox21 classification with GATv2.ipynb
│   ├── Tox21 classification with GATv2_ver_2.ipynb
│   ├── train_esol.ipynb
│   └── tox21_gnn_comparison_scaffold.csv
│
└── 📁 data/                    # Auto-generated dataset cache
```

---

## 🔧 Technical Details

### Molecular Featurization

**Node Features (9 dimensions):**
| Feature | Description | Range |
|---------|-------------|-------|
| Atomic Number | Element type | 1-118 |
| Chirality | Stereochemistry | 0-3 |
| Formal Charge | Ionic charge | -2 to +2 |
| Explicit Hs | Explicit hydrogens | 0-4 |
| Hybridization | sp, sp2, sp3, etc. | 0-5 |
| Aromaticity | Is aromatic? | 0-1 |
| In Ring | Part of ring? | 0-1 |
| Radical Electrons | Unpaired electrons | 0-2 |
| Degree | Number of bonds | 0-6 |

**Edge Features (3 dimensions):**
| Feature | Description |
|---------|-------------|
| Single Bond | Is single bond? (0/1) |
| Double Bond | Is double bond? (0/1) |
| Triple Bond | Is triple bond? (0/1) |

### Training Configuration

```python
{
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "epochs": 100,
    "early_stopping_patience": 15,
    "dropout": 0.25,
    "hidden_channels": 256,
    "num_layers": 4,
    "pooling": "mean + sum",
    "split": "scaffold"
}
```

---

## 🌐 Deployment

### Hugging Face Spaces (Current Deployment)

The app is deployed at: [https://huggingface.co/spaces/Epion09g/MolGNN-Tox21-Predictor](https://huggingface.co/spaces/Epion09g/MolGNN-Tox21-Predictor)

### Docker Deployment

```bash
# Build image
docker build -t molgnn-tox21 .

# Run container
docker run -p 7860:7860 molgnn-tox21
```

### Local Development

```bash
python app.py
# Access at http://localhost:7860
```

---

## 👥 Team

This project was developed as part of an academic research initiative on **Graph ML-Enabled Molecular Design** at the intersection of machine learning and computational chemistry.

### Contributors

| Name | Role |
|------|------|
| **Saptarshi Ghosh** | Team Lead & Research Coordinator |
| **Sumit Chaira** | UI/Deployment & Visualization Developer |
| **Mangaldip Dhua** | Data Engineer & Preprocessing Specialist |
| **Uday Shankar Dey** | GNN Model Developer |
| **Arnab Subhra Ghosh** | Model Evaluation & Optimization Engineer |

---

## 📚 References

### Papers

1. **Tox21 Challenge**: Huang, R., et al. "Tox21Challenge to Build Predictive Models of Nuclear Receptor and Stress Response Pathways." *Frontiers in Environmental Science* (2016).

2. **MoleculeNet**: Wu, Z., et al. "MoleculeNet: A Benchmark for Molecular Machine Learning." *Chemical Science* (2018). [arXiv:1703.00564](https://arxiv.org/abs/1703.00564)

3. **GIN**: Xu, K., et al. "How Powerful are Graph Neural Networks?" *ICLR* (2019). [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)

4. **GATv2**: Brody, S., et al. "How Attentive are Graph Attention Networks?" *ICLR* (2022). [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)

### Libraries

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - GNN library
- [RDKit](https://www.rdkit.org/) - Cheminformatics toolkit
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [3Dmol.js](https://3dmol.csb.pitt.edu/) - Molecular visualization

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 Acknowledgments

- **Tox21 Challenge** organizers for the benchmark dataset
- **PyTorch Geometric** team for the excellent GNN library
- **RDKit** developers for cheminformatics tools
- **Hugging Face** for free hosting on Spaces

---

<div align="center">

### ⭐ Star this repo if you find it useful!

**Made with ❤️ for computational chemistry and drug discovery**

[![GitHub stars](https://img.shields.io/github/stars/eliot-99/Graph-ML-Enabled-Molecular-Design-Assistant-using-Graph-Neural-Networks?style=social)](https://github.com/eliot-99/Graph-ML-Enabled-Molecular-Design-Assistant-using-Graph-Neural-Networks)

</div>
