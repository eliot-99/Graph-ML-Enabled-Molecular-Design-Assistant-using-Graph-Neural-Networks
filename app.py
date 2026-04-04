import os
import sys
from datetime import datetime

# Patch for Windows RDKit DLL issue
# This creates a fake inchi module to bypass the blocked DLL
class FakeInchi:
    pass

sys.modules['rdkit.Chem.inchi'] = FakeInchi()

import base64
import io
from flask import Flask, render_template, request, jsonify, Response
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import pandas as pd
import numpy as np
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TASK_NAMES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

# ==================== MODEL CLASSES (exact match to your training) ====================
from torch_geometric.nn import GINEConv, GCNConv, GATv2Conv, global_mean_pool, global_add_pool, BatchNorm
from torch.nn import Sequential, Linear, ReLU, Dropout

class BestTox21GNN(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden=256, num_tasks=12, dropout=0.25):
        super().__init__()
        def gin_mlp(in_dim, out_dim):
            return Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
        self.conv1 = GINEConv(gin_mlp(in_channels, hidden), edge_dim=edge_dim)
        self.bn1 = BatchNorm(hidden)
        self.conv2 = GINEConv(gin_mlp(hidden, hidden), edge_dim=edge_dim)
        self.bn2 = BatchNorm(hidden)
        self.conv3 = GINEConv(gin_mlp(hidden, hidden), edge_dim=edge_dim)
        self.bn3 = BatchNorm(hidden)
        self.conv4 = GINEConv(gin_mlp(hidden, hidden), edge_dim=edge_dim)
        self.bn4 = BatchNorm(hidden)
        self.lin1 = Linear(hidden * 2, hidden)
        self.lin2 = Linear(hidden, num_tasks)
        self.dropout = dropout

    def forward(self, data):
        x = data.x.float().clamp(-10, 10)
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn4(self.conv4(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_pool = torch.cat([x_mean, x_sum], dim=-1)
        x = F.relu(self.lin1(x_pool))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).clamp(-10, 10)

class GCNTox21(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, num_tasks=12, dropout=0.25):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden); self.bn1 = BatchNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden); self.bn2 = BatchNorm(hidden)
        self.conv3 = GCNConv(hidden, hidden); self.bn3 = BatchNorm(hidden)
        self.conv4 = GCNConv(hidden, hidden); self.bn4 = BatchNorm(hidden)
        self.lin1 = Linear(hidden * 2, hidden)
        self.lin2 = Linear(hidden, num_tasks)
        self.dropout = dropout

    def forward(self, data):
        x = data.x.float().clamp(-10, 10)
        edge_index = data.edge_index
        batch = data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_pool = torch.cat([x_mean, x_sum], dim=-1)
        x = F.relu(self.lin1(x_pool))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).clamp(-10, 10)

class GATv2Tox21(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, num_tasks=12, dropout=0.3, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden//heads, heads=heads, concat=True, dropout=dropout); self.bn1 = BatchNorm(hidden)
        self.conv2 = GATv2Conv(hidden, hidden//heads, heads=heads, concat=True, dropout=dropout); self.bn2 = BatchNorm(hidden)
        self.conv3 = GATv2Conv(hidden, hidden//heads, heads=heads, concat=True, dropout=dropout); self.bn3 = BatchNorm(hidden)
        self.conv4 = GATv2Conv(hidden, hidden//heads, heads=heads, concat=True, dropout=dropout); self.bn4 = BatchNorm(hidden)
        self.lin1 = Linear(hidden * 2, hidden)
        self.lin2 = Linear(hidden, num_tasks)
        self.dropout = dropout

    def forward(self, data):
        x = data.x.float().clamp(-10, 10)
        edge_index = data.edge_index
        batch = data.batch
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn4(self.conv4(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_pool = torch.cat([x_mean, x_sum], dim=-1)
        x = F.elu(self.lin1(x_pool))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).clamp(-10, 10)

# ==================== LOAD DATASET + NORMALIZERS (edge_dim fixed to 3) ====================
print("Loading dataset and normalizers...")
from torch_geometric.datasets import MoleculeNet
tox21 = MoleculeNet(root='./data/MoleculeNet', name='Tox21')

# Node normalization (unchanged)
all_x = torch.cat([d.x for d in tox21], dim=0).float()
mean_x = all_x.mean(dim=0)
std_x = all_x.std(dim=0) + 1e-6

# FORCE edge_dim=3 to match your saved checkpoint (this was the cause of the error)
edge_dim = 3

# Edge normalization - use only first 3 features to match your training checkpoint
valid_edge_attrs = []
for d in tox21:
    if hasattr(d, 'edge_attr') and d.edge_attr is not None:
        ea = d.edge_attr.float()[:, :3]          # slice to 3 dims
        valid_edge_attrs.append(ea)
all_edge = torch.cat(valid_edge_attrs, dim=0)
mean_e = all_edge.mean(dim=0)
std_e = all_edge.std(dim=0) + 1e-6

print(f"Using edge_dim = {edge_dim} (matches your checkpoint)")

# ==================== LOAD MODELS ====================
model_gnn = BestTox21GNN(in_channels=9, edge_dim=edge_dim, hidden=256, num_tasks=12, dropout=0.25).to(device)
model_gnn.load_state_dict(torch.load("models/best_gnn_tox21_scaffold.pt", map_location=device))
model_gnn.eval()

model_gcn = GCNTox21(in_channels=9, hidden=256, num_tasks=12, dropout=0.25).to(device)
model_gcn.load_state_dict(torch.load("models/best_gcn_tox21_scaffold.pt", map_location=device))
model_gcn.eval()

model_gat = GATv2Tox21(in_channels=9, hidden=256, num_tasks=12, dropout=0.3, heads=4).to(device)
model_gat.load_state_dict(torch.load("models/best_gatv2_tox21_scaffold.pt", map_location=device))
model_gat.eval()

print("✅ All 3 models loaded successfully!")

# ==================== SMILES → PyG Data (3 edge features to match checkpoint) ====================
def smiles_to_pyg_data(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features (9 dims - same as MoleculeNet)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(), int(atom.GetChiralTag()), atom.GetFormalCharge(),
            atom.GetNumExplicitHs(), int(atom.GetHybridization()),
            int(atom.GetIsAromatic()), int(atom.IsInRing()),
            atom.GetNumRadicalElectrons(), atom.GetDegree()
        ])
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge features - EXACTLY 3 dims (matches your training checkpoint)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        bt = bond.GetBondType()
        feat = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE
        ]
        edge_attr.extend([feat, feat])

    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    return data

# ==================== NORMALIZE (slice edge_attr to 3 dims) ====================
def normalize_data(data):
    data.x = ((data.x.float() - mean_x) / std_x).clamp(-10.0, 10.0)
    data.x = torch.nan_to_num(data.x, nan=0.0)

    if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.shape[1] > 0:
        ea = data.edge_attr.float()[:, :3]                     # force 3 dims
        data.edge_attr = ((ea - mean_e) / std_e).clamp(-10.0, 10.0)
        data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)
    return data

# ==================== PREDICTION ====================
def run_prediction(model, data):
    data = normalize_data(data).to(device)
    if data.batch is None:                                      # single molecule fix
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs.tolist()


def build_export_rows(results):
    rows = []
    for item in results:
        smiles = item.get("smiles", "")
        model_name = item.get("model", "")
        probs = item.get("probs", [])

        for i, prob in enumerate(probs):
            endpoint = TASK_NAMES[i] if i < len(TASK_NAMES) else f"TASK_{i+1}"
            prediction = "TOXIC" if float(prob) > 0.5 else "SAFE"
            rows.append({
                "SMILES": smiles,
                "Model": model_name,
                "Endpoint": endpoint,
                "Probability": float(prob),
                "Prediction": prediction,
            })
    return rows

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_choice = request.form.get("model")
        smiles_str = request.form.get("smiles", "").strip()
        csv_file = request.files.get("csv")

        if not model_choice:
            return jsonify({"error": "No model selected"}), 400

        results = []
        response = {}

        MODEL_MAP = {
            "gnn":   (model_gnn,   "GINE"),
            "gcn":   (model_gcn,   "GCN"),
            "gatv2": (model_gat,   "GATv2"),
        }

        if smiles_str:
            data = smiles_to_pyg_data(smiles_str)
            if data is None:
                return jsonify({"error": "Invalid SMILES string"}), 400

            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return jsonify({"error": "Could not parse molecule"}), 400

            # 2D image
            AllChem.Compute2DCoords(mol)
            img_buffer = io.BytesIO()
            Draw.MolToImage(mol, size=(520, 520)).save(img_buffer, format="PNG")
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

            # 3D SDF
            mol3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol3d, randomSeed=42) != 0:
                AllChem.EmbedMolecule(mol3d, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(mol3d)
            sdf_block = Chem.MolToMolBlock(mol3d)

            if model_choice == "all":
                for key, (mdl, name) in MODEL_MAP.items():
                    probs = run_prediction(mdl, data.clone())
                    results.append({"smiles": smiles_str, "model": name, "probs": probs})
                response = {"type": "batch", "results": results}
            else:
                if model_choice not in MODEL_MAP:
                    return jsonify({"error": "Unknown model"}), 400
                mdl, name = MODEL_MAP[model_choice]
                probs = run_prediction(mdl, data.clone())
                results = [{"smiles": smiles_str, "model": name, "probs": probs}]
                response = {
                    "type": "single",
                    "image": img_b64,
                    "sdf": sdf_block,
                    "probs": probs,
                    "results": results
                }

        elif csv_file:
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                return jsonify({"error": f"Cannot read CSV: {str(e)}"}), 400

            if "SMILES" not in df.columns:
                return jsonify({"error": "CSV must contain 'SMILES' column"}), 400

            selected_model = MODEL_MAP.get(model_choice)
            if model_choice != "all" and not selected_model:
                return jsonify({"error": "Invalid model for batch"}), 400

            for sm in df["SMILES"].dropna().astype(str):
                sm = sm.strip()
                if not sm: continue
                d = smiles_to_pyg_data(sm)
                if d is None: continue

                if model_choice == "all":
                    for key, (mdl, name) in MODEL_MAP.items():
                        probs = run_prediction(mdl, d.clone())
                        results.append({"smiles": sm, "model": name, "probs": probs})
                else:
                    mdl, name = selected_model
                    probs = run_prediction(mdl, d.clone())
                    results.append({"smiles": sm, "model": name, "probs": probs})

            response = {"type": "batch", "results": results}

        else:
            return jsonify({"error": "No SMILES or CSV provided"}), 400

        app.config["LAST_EXPORT_ROWS"] = build_export_rows(results)

        return jsonify(response)

    except Exception as e:
        import traceback
        print("Predict endpoint error:\n" + traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/export_csv", methods=["GET"])
def export_csv():
    rows = app.config.get("LAST_EXPORT_ROWS", [])
    if not rows:
        return jsonify({"error": "No prediction results available to export."}), 400

    df = pd.DataFrame(rows)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    filename = f"tox21_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


if __name__ == "__main__":
    # Use port 7860 for Hugging Face Spaces, fallback to 5000 for local
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)