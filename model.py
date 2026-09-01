"""Hybrid Swin Transformer + quantum-inspired bottleneck.

The three bottleneck modes exist so the ablation is a one-line config change:

    "quantum"   Swin -> FC(768->n) + tanh -> n-qubit circuit -> head
    "classical" Swin -> FC(768->n) + tanh -> head          (matched control)
    "none"      Swin -> head                                (backbone only)

The "classical" arm is the control that matters. It has the identical
bottleneck width, so if the quantum arm does not beat it, the gain is from
compression rather than from the entangling structure, and the paper's central
claim does not hold. Report that outcome if you get it.

Terminology: the circuit is simulated on classical hardware via PennyLane's
default.qubit. No quantum-hardware claim is made anywhere in this work. State
this explicitly in the manuscript.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# quantum-inspired layer
# ---------------------------------------------------------------------------

def build_quantum_layer(n_qubits: int, n_layers: int) -> nn.Module:
    """AngleEmbedding -> BasicEntanglerLayers x L -> Pauli-Z expectations."""
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits)})


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------

class HybridSwin(nn.Module):
    def __init__(self, cfg, n_classes: int = 3, pretrained: bool = True):
        super().__init__()
        import timm

        self.cfg = cfg
        self.mode = cfg.bottleneck
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=pretrained, num_classes=0)
        feat_dim = getattr(self.backbone, "num_features", cfg.embed_dim)

        if self.mode == "none":
            self.proj = nn.Identity()
            self.quantum = None
            head_in = feat_dim
        else:
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, cfg.n_qubits),
                nn.Tanh(),          # bounds inputs to [-1,1] for angle encoding
            )
            self.quantum = (build_quantum_layer(cfg.n_qubits, cfg.n_layers)
                            if self.mode == "quantum" else None)
            head_in = cfg.n_qubits

        self.head = nn.Linear(head_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.proj(z)
        if self.quantum is not None:
            z = self.quantum(z).float()
        return self.head(z)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    q = (sum(p.numel() for p in model.quantum.parameters())
         if getattr(model, "quantum", None) is not None else 0)
    return {"total": total, "quantum": q, "classical": total - q}
