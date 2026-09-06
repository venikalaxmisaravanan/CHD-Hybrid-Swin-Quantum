"""Hybrid Swin Transformer + quantum-inspired bottleneck.

Original bottleneck design used for the original swin_only, classical4,
and q4_L1 experiments.

This version contains no BatchNorm and no angle scaling.

Three bottleneck modes:

    "quantum"   Swin -> FC(768->n) -> tanh -> circuit -> head
    "classical" Swin -> FC(768->n) -> tanh -> head
    "none"      Swin -> head

The quantum circuit is simulated classically using PennyLane default.qubit.
No quantum hardware is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_quantum_layer(n_qubits: int, n_layers: int) -> nn.Module:
    """AngleEmbedding -> BasicEntanglerLayers x L -> Pauli-Z expectations."""

    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(
            inputs,
            wires=range(n_qubits),
            rotation="X"
        )
        qml.BasicEntanglerLayers(
            weights,
            wires=range(n_qubits)
        )
        return [
            qml.expval(qml.PauliZ(w))
            for w in range(n_qubits)
        ]

    return qml.qnn.TorchLayer(
        circuit,
        {"weights": (n_layers, n_qubits)}
    )


class HybridSwin(nn.Module):

    def __init__(
        self,
        cfg,
        n_classes: int = 3,
        pretrained: bool = True
    ):
        super().__init__()

        import timm

        self.cfg = cfg
        self.mode = cfg.bottleneck

        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=pretrained,
            num_classes=0
        )

        feat_dim = getattr(
            self.backbone,
            "num_features",
            cfg.embed_dim
        )

        if self.mode == "none":

            self.proj = nn.Identity()
            self.quantum = None
            head_in = feat_dim

        else:

            self.proj = nn.Sequential(
                nn.Linear(
                    feat_dim,
                    cfg.n_qubits
                ),
                nn.Tanh(),
            )

            self.quantum = (
                build_quantum_layer(
                    cfg.n_qubits,
                    cfg.n_layers
                )
                if self.mode == "quantum"
                else None
            )

            head_in = cfg.n_qubits

        self.head = nn.Linear(
            head_in,
            n_classes
        )

    def quantum_parameters(self):

        return (
            list(self.quantum.parameters())
            if self.quantum is not None
            else []
        )

    def to(self, *args, **kwargs):

        out = super().to(*args, **kwargs)

        if out.quantum is not None:
            out.quantum.to("cpu")

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.backbone(x)
        z = self.proj(z)

        if self.quantum is not None:

            dev = z.device

            z = self.quantum(
                z.to("cpu")
            ).float().to(dev)

        return self.head(z)


def count_params(model: nn.Module) -> dict:

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    q = sum(
        p.numel()
        for p in model.quantum_parameters()
    )

    return {
        "total": total,
        "quantum": q,
        "classical": total - q
    }
