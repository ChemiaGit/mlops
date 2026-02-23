"""
Standalone PyTorch implementation of DeepChem's CGCNNModel.

Reconstructed to match the exact state dict key names produced by
dc.models.CGCNNModel so that checkpoints load without DeepChem at
inference time.

State dict keys (from checkpoint5.pt inspection):
  embedding.weight              (hidden_node_dim, node_dim)
  embedding.bias                (hidden_node_dim,)
  conv_layers.{i}.linear.weight (2*hidden_node_dim, 2*hidden_node_dim + edge_dim)
  conv_layers.{i}.linear.bias   (2*hidden_node_dim,)
  conv_layers.{i}.batch_norm.*  (2*hidden_node_dim,)
  fc.weight                     (predictor_hidden_feats, hidden_node_dim)
  fc.bias                       (predictor_hidden_feats,)
  out.weight                    (1, predictor_hidden_feats)
  out.bias                      (1,)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGCNNLayer(nn.Module):
    """
    Single crystal-graph convolution layer (DeepChem-compatible).

    For each edge (i → j) with feature e_ij:
        z = linear( cat(h_i, h_j, e_ij) )       shape (2*H,)
        z = batch_norm(z)
        σ = sigmoid(z[:H])
        π = softplus(z[H:])
        msg = σ * π
    Then scatter-add msg into source atom i (residual update).
    """

    def __init__(self, hidden_node_dim: int, edge_dim: int):
        super().__init__()
        self.H = hidden_node_dim
        self.linear = nn.Linear(2 * hidden_node_dim + edge_dim, 2 * hidden_node_dim)
        self.batch_norm = nn.BatchNorm1d(2 * hidden_node_dim)

    def forward(
        self,
        node_feats: torch.Tensor,   # (N, H)
        edge_index: torch.Tensor,   # (2, E)  [src, dst]
        edge_feats: torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:              # (N, H)
        src = edge_index[0]  # center atoms
        dst = edge_index[1]  # neighbor atoms

        z = torch.cat([node_feats[src], node_feats[dst], edge_feats], dim=1)
        z = self.linear(z)
        z = self.batch_norm(z)

        sigma = torch.sigmoid(z[:, : self.H])
        pi    = F.softplus(z[:, self.H :])
        msg   = sigma * pi  # (E, H)

        # Scatter-add messages to source atoms
        agg = torch.zeros_like(node_feats)
        agg.scatter_add_(0, src.unsqueeze(1).expand_as(msg), msg)

        return node_feats + agg  # residual


class CGCNNTorchModel(nn.Module):
    """
    Full CGCNN graph neural network (DeepChem-compatible).

    Forward pass:
      1. Linear embedding of atom features
      2. num_conv convolution layers
      3. Mean-pool atom features per crystal
      4. Linear + softplus → output head
    """

    def __init__(
        self,
        node_dim: int,
        hidden_node_dim: int,
        edge_dim: int,
        num_conv: int,
        predictor_hidden_feats: int,
    ):
        super().__init__()
        self.embedding = nn.Linear(node_dim, hidden_node_dim)
        self.conv_layers = nn.ModuleList(
            [CGCNNLayer(hidden_node_dim, edge_dim) for _ in range(num_conv)]
        )
        self.fc  = nn.Linear(hidden_node_dim, predictor_hidden_feats)
        self.out = nn.Linear(predictor_hidden_feats, 1)

    def forward(
        self,
        node_feats: torch.Tensor,          # (N_total, node_dim)
        edge_index: torch.Tensor,          # (2, E_total)
        edge_feats: torch.Tensor,          # (E_total, edge_dim)
        crystal_atom_idx: list[torch.Tensor],  # list of 1-D index tensors, one per crystal
    ) -> torch.Tensor:                     # (batch, 1)
        h = self.embedding(node_feats)     # (N_total, H)

        for layer in self.conv_layers:
            h = layer(h, edge_index, edge_feats)

        # Mean-pool atoms per crystal
        pooled = torch.stack(
            [h[idx].mean(dim=0) for idx in crystal_atom_idx]
        )  # (batch, H)

        out = F.softplus(self.fc(pooled))  # (batch, predictor_hidden_feats)
        return self.out(out)               # (batch, 1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def infer_params(state_dict: dict) -> dict:
    """
    Read model hyperparameters directly from checkpoint weight shapes.
    No need to store them separately.
    """
    hidden_node_dim = state_dict["embedding.weight"].shape[0]
    node_dim        = state_dict["embedding.weight"].shape[1]

    # conv input = hidden_node_dim (h_i) + hidden_node_dim (h_j) + edge_dim
    conv_in = state_dict["conv_layers.0.linear.weight"].shape[1]
    edge_dim = conv_in - 2 * hidden_node_dim

    num_conv = sum(
        1 for k in state_dict
        if k.startswith("conv_layers.") and k.endswith(".linear.weight")
    )

    predictor_hidden_feats = state_dict["fc.weight"].shape[0]

    return {
        "node_dim":               node_dim,
        "hidden_node_dim":        hidden_node_dim,
        "edge_dim":               edge_dim,
        "num_conv":               num_conv,
        "predictor_hidden_feats": predictor_hidden_feats,
    }


def load_from_checkpoint(checkpoint_path: str, device: torch.device) -> CGCNNTorchModel:
    """Load a DeepChem-saved CGCNN checkpoint into a CGCNNTorchModel."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise ValueError(f"Unrecognised checkpoint format in {checkpoint_path}")

    params = infer_params(state_dict)
    model = CGCNNTorchModel(**params).to(device)
    model.load_state_dict(state_dict)
    return model
