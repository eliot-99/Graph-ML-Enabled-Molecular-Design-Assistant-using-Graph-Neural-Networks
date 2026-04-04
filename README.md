# 🧬 MolGNN Tox21 Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A Graph Neural Network-powered web application for predicting molecular toxicity across 12 biological endpoints**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Deployment](#-deployment)

</div>

---

## 📖 Overview

MolGNN Tox21 Predictor is an AI-powered molecular toxicity prediction tool that leverages state-of-the-art Graph Neural Networks (GNNs) to assess the potential toxicity of chemical compounds. The system predicts toxicity across all 12 endpoints of the **Tox21 benchmark dataset**, which is widely used in drug discovery and chemical safety assessment.

### 🎯 Tox21 Endpoints Predicted

| Endpoint | Description |
|----------|-------------|
| NR-AR | Nuclear Receptor - Androgen Receptor |
| NR-AR-LBD | Nuclear Receptor - Androgen Receptor Ligand Binding Domain |
| NR-AhR | Nuclear Receptor - Aryl Hydrocarbon Receptor |
| NR-Aromatase | Nuclear Receptor - Aromatase |
| NR-ER | Nuclear Receptor - Estrogen Receptor |
| NR-ER-LBD | Nuclear Receptor - Estrogen Receptor Ligand Binding Domain |
| NR-PPAR-gamma | Nuclear Receptor - Peroxisome Proliferator-Activated Receptor Gamma |
| SR-ARE | Stress Response - Antioxidant Response Element |
| SR-ATAD5 | Stress Response - ATAD5 |
| SR-HSE | Stress Response - Heat Shock Element |
| SR-MMP | Stress Response - Mitochondrial Membrane Potential |
| SR-p53 | Stress Response - p53 |

---

## ✨ Features

- 🔬 **Multi-Model Ensemble**: Choose from 3 GNN architectures (GINE, GCN, GATv2) or use all models together
- 🎨 **Interactive 2D/3D Visualization**: View molecules in 2D and interactive 3D with 3Dmol.js
- 📊 **Comprehensive Results**: Toxicity probabilities for all 12 Tox21 endpoints
- 📁 **Batch Processing**: Upload CSV files for bulk predictions
- 💾 **Export Results**: Download predictions as CSV for further analysis
- 🌐 **Modern Web UI**: Responsive dark-themed interface built with Bootstrap 5

---

## 🖼️ Demo

### Input Interface
- Enter SMILES strings directly or upload CSV files
- Select from individual models or ensemble prediction

### Results Display
- Color-coded toxicity predictions (Safe/Toxic)
- Interactive 3D molecular visualization
- Downloadable CSV reports

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, for faster inference)

### Clone the Repository

```bash
git clone https://github.com/yourusername/MolGNN-Tox21-Predictor.git
cd MolGNN-Tox21-Predictor
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: PyTorch Geometric installation may require additional steps. Follow the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
# For PyTorch Geometric (after installing PyTorch)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

---

## 🚀 Usage

### Start the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Using the Web Interface

1. **Single Molecule Prediction**
   - Enter a SMILES string (e.g., `CCO` for Ethanol, `CC(=O)Oc1ccccc1C(=O)O` for Aspirin)
   - Select a model (GINE, GCN, GATv2, or All)
   - Click "Predict"

2. **Batch Prediction**
   - Prepare a CSV file with a column named `SMILES`
   - Upload the file and select model(s)
   - Download results as CSV

### Example SMILES

```
CCO                           # Ethanol
CC(=O)Oc1ccccc1C(=O)O         # Aspirin
Cn1cnc2c1c(=O)n(c(=O)n2C)C    # Caffeine
CC(C)Cc1ccc(cc1)C(C)C(=O)O    # Ibuprofen
```

---

## 🧠 Models

### Architecture Overview

This project implements three Graph Neural Network architectures:

| Model | Architecture | Key Features |
|-------|-------------|--------------|
| **GINE** | Graph Isomorphism Network with Edge features | Best overall performance, uses edge attributes |
| **GCN** | Graph Convolutional Network | Classical architecture, efficient training |
| **GATv2** | Graph Attention Network v2 | Dynamic attention mechanism |

### Model Performance (Scaffold Split)

| Model | Macro AUC-ROC | Parameters |
|-------|---------------|------------|
| GINE | ~0.78 | 256 hidden |
| GCN | ~0.75 | 256 hidden |
| GATv2 | ~0.76 | 256 hidden, 4 heads |

### Training Details

- **Dataset**: Tox21 (MoleculeNet)
- **Split**: Scaffold Split (recommended for realistic evaluation)
- **Node Features**: 9 atomic features (atomic number, chirality, formal charge, etc.)
- **Edge Features**: 3 bond features (single, double, triple bonds)
- **Pooling**: Combined mean + sum pooling

---

## 📁 Project Structure

```
MolGNN-Tox21-Predictor/
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── .gitattributes           # Git LFS tracking for model files
├── Procfile                 # Deployment config (Heroku/Render)
├── runtime.txt              # Python version specification
├── LICENSE                  # MIT License
├── models/                  # Pre-trained model weights
│   ├── best_gnn_tox21_scaffold.pt   (GINE model)
│   ├── best_gcn_tox21_scaffold.pt   (GCN model)
│   └── best_gatv2_tox21_scaffold.pt (GATv2 model)
├── templates/               # HTML templates
│   ├── index.html          # Main prediction interface
│   └── about.html          # About page with team info
├── static/                  # Static assets
│   └── team/               # Team member photos
├── data/                    # Auto-generated dataset cache
└── notebooks/               # Training notebooks & results
    ├── GAT_Tox21.ipynb
    ├── Tox21 classification with GATv2.ipynb
    ├── Tox21 classification with GATv2_ver_2.ipynb
    ├── train_esol.ipynb
    ├── tox21_gnn_comparison_scaffold.csv
    └── figures/             # 29 training result plots
```

---

## 📓 Notebooks

The `notebooks/` directory contains model training experiments:

| Notebook | Description |
|----------|-------------|
| `GAT_Tox21.ipynb` | Graph Attention Network training on Tox21 |
| `Tox21 classification with GATv2.ipynb` | GATv2 architecture experiments |
| `Tox21 classification with GATv2_ver_2.ipynb` | Improved GATv2 with scaffold split |
| `train_esol.ipynb` | ESOL solubility prediction (GCN) |

### Training Result Figures (29 plots)

The `notebooks/figures/` directory includes:
- Training/validation loss curves
- ROC-AUC comparison plots
- t-SNE chemical space visualizations
- Per-task performance analysis

---

## 🌐 Deployment

### Hugging Face Spaces (Recommended - Free)

1. Create a new Space at [Hugging Face Spaces](https://huggingface.co/spaces)
2. Select "Gradio" or "Docker" SDK
3. Upload your files or connect your GitHub repo
4. The app will deploy automatically

### Render (Free Tier)

1. Create a `render.yaml`:
```yaml
services:
  - type: web
    name: molgnn-tox21
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
```

2. Connect your GitHub repo to Render
3. Deploy automatically on push

### Railway (Free Tier)

1. Install Railway CLI: `npm i -g @railway/cli`
2. Run `railway login` and `railway init`
3. Deploy with `railway up`

### PythonAnywhere (Free Tier)

1. Create a free account at [PythonAnywhere](https://www.pythonanywhere.com)
2. Upload your files via the web interface
3. Configure WSGI for Flask

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t molgnn-tox21 .
docker run -p 5000:5000 molgnn-tox21
```

---

## 📚 References

- **Tox21 Dataset**: [Tox21 Data Challenge](https://tripod.nih.gov/tox21/challenge/)
- **MoleculeNet**: [Paper](https://arxiv.org/abs/1703.00564)
- **PyTorch Geometric**: [Documentation](https://pytorch-geometric.readthedocs.io/)
- **RDKit**: [Documentation](https://www.rdkit.org/docs/)

---

## 👥 Team

This project was developed as part of an academic research initiative on Graph ML-Enabled Molecular Design.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Tox21 Challenge organizers for the dataset
- PyTorch Geometric team for the excellent GNN library
- RDKit developers for cheminformatics tools

---

<div align="center">

**Made with ❤️ for computational chemistry and drug discovery**

</div>
