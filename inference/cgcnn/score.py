"""
Azure ML scoring script for CGCNN property prediction models.

Models were trained with DeepChem's CGCNNModel (dc.models.CGCNNModel).
This script loads the checkpoint without DeepChem by reconstructing the
exact same PyTorch architecture from the state dict key shapes.

Input JSON (single):
    {"cif": "<cif_text>", "id": "optional_id"}

Input JSON (batch):
    {"structures": [{"cif": "<cif_text>", "id": "struct_001"}, ...]}

Output JSON:
    {"id": "struct_001", "prediction": 0.234, "property": "specific_heat"}
    or
    {"predictions": [{"id": "...", "prediction": 0.234, "property": "..."}, ...]}
"""

import glob
import json
import logging
import os
import re

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Globals loaded once in init()
model = None
device = None
property_name = None

# Must match dc.feat.CGCNNFeaturizer defaults
RADIUS       = 8.0
MAX_NUM_NBR  = 12
GAUSSIAN_DMIN = 0.0
GAUSSIAN_STEP = 0.2


# ---------------------------------------------------------------------------
# Azure ML entrypoints
# ---------------------------------------------------------------------------

def init():
    global model, device, property_name

    from cgcnn_model import load_from_checkpoint

    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  model dir: {model_dir}")

    checkpoint_path = _find_checkpoint(model_dir)
    logger.info(f"Loading: {checkpoint_path}")

    model = load_from_checkpoint(checkpoint_path, device)
    model.eval()

    property_name = os.environ.get(
        "MODEL_PROPERTY", _infer_property_name(model_dir)
    )
    logger.info(f"Ready — property: {property_name}")


def run(raw_data: str) -> str:
    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    if "structures" in data:
        items = data["structures"]
    elif "cif" in data:
        items = [{"cif": data["cif"], "id": data.get("id", "input_0")}]
    else:
        return json.dumps({"error": "Input must contain 'cif' or 'structures'"})

    results = []
    for item in items:
        sid = item.get("id", "unknown")
        try:
            pred = _predict_single(item["cif"])
            results.append({"id": sid, "prediction": pred, "property": property_name})
        except Exception as e:
            logger.error(f"Error on {sid}: {e}")
            results.append({"id": sid, "error": str(e)})

    return json.dumps(results[0] if len(results) == 1 else {"predictions": results})


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict_single(cif_text: str) -> float:
    from pymatgen.core import Structure

    structure = Structure.from_str(cif_text, fmt="cif")
    node_feats, edge_index, edge_feats, crystal_atom_idx = _featurize(structure)

    node_t   = torch.tensor(node_feats, dtype=torch.float32).to(device)
    edge_i_t = torch.tensor(edge_index, dtype=torch.long).to(device)
    edge_f_t = torch.tensor(edge_feats, dtype=torch.float32).to(device)
    idx_t    = [torch.tensor(idx, dtype=torch.long).to(device)
                for idx in crystal_atom_idx]

    with torch.no_grad():
        out = model(node_t, edge_i_t, edge_f_t, idx_t)

    return float(out.squeeze())


# ---------------------------------------------------------------------------
# Featurizer  (matches dc.feat.CGCNNFeaturizer defaults exactly)
# ---------------------------------------------------------------------------

def _featurize(structure):
    """
    pymatgen Structure → (node_feats, edge_index, edge_feats, crystal_atom_idx)

    Matches dc.feat.CGCNNFeaturizer(radius=8.0, max_neighbors=12, step=0.2).
    """
    N = len(structure)
    node_feats = np.vstack([_atom_features(site.specie.Z) for site in structure])

    all_nbrs = structure.get_all_neighbors(RADIUS, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    src_list, dst_list, dist_list = [], [], []
    for i, nbrs in enumerate(all_nbrs):
        actual = nbrs[:MAX_NUM_NBR]
        pad    = MAX_NUM_NBR - len(actual)

        for nbr in actual:
            src_list.append(i)
            dst_list.append(int(nbr[2]))
            dist_list.append(float(nbr[1]))

        # Pad short neighbor lists with self-loops at max distance
        for _ in range(pad):
            src_list.append(i)
            dst_list.append(i)
            dist_list.append(RADIUS + 1.0)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)    # (2, E)
    dists      = np.array(dist_list, dtype=np.float32)             # (E,)
    edge_feats = _gaussian_expand(dists)                           # (E, G)
    crystal_atom_idx = [list(range(N))]

    return node_feats, edge_index, edge_feats, crystal_atom_idx


def _atom_features(atomic_number: int) -> np.ndarray:
    """92-dim one-hot by atomic number. Matches CGCNNFeaturizer default."""
    fea = np.zeros(92, dtype=np.float32)
    fea[min(atomic_number - 1, 91)] = 1.0
    return fea


def _gaussian_expand(distances: np.ndarray) -> np.ndarray:
    """Gaussian basis expansion matching dc.feat.CGCNNFeaturizer: 41 filter centers."""
    centers = np.arange(GAUSSIAN_DMIN, RADIUS + GAUSSIAN_STEP, GAUSSIAN_STEP)
    var     = GAUSSIAN_STEP
    return np.exp(-((distances[:, np.newaxis] - centers) ** 2) / var ** 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint(model_dir: str) -> str:
    candidates = glob.glob(os.path.join(model_dir, "**/*.pt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .pt file found under {model_dir}")
    # Prefer highest-numbered checkpoint
    candidates.sort(key=lambda p: int(m.group(1)) if (m := re.search(r'(\d+)', os.path.basename(p))) else 0)
    return candidates[-1]


def _infer_property_name(model_dir: str) -> str:
    for part in reversed(model_dir.split("/")):
        if part.upper().startswith("CGCNN_"):
            return re.sub(r'_\d{10,}$', '', part)[len("CGCNN_"):].lower()
    return "unknown"
