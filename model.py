"""Hybrid Swin Transformer + quantum-inspired bottleneck.

Three bottleneck modes so the ablation is a one-line config change:

    "quantum"   Swin -> FC(768->n) -> BN -> tanh -> x angle_scale -> circuit -> head
    "classical" Swin -> FC(768->n) -> BN -> tanh -> head        (matched control)
    "none"      Swin -> head                                     (backbone only)

The "classical" arm is the control that matters. Identical bottleneck width and
identical pre-circuit stack, so if the quantum arm does not beat it, the gain is
from compression rather than entangling structure. Report that if you get it.

Terminology: the circuit is simulated classically via PennyLane's default.qubit.
No quantum-hardware claim is made anywhere in this work.

DEVICE NOTE: default.qubit builds its state vector on CPU and does not follow a
torch model onto CUDA. The quantum layer is pinned to CPU and the bottleneck
vector crosses that boundary in forward(). Autograd tracks the transfer.

GRADIENT NOTE: diff_method="backprop" differentiates through the simulator. For
a simulated circuit this gives gradients mathematically identical to the
parameter-shift rule, and is much faster. Parameter-shift would be required on
physical hardware. Section 3.3 of the manuscript must state what was used.

ANGLE-SCALE NOTE (why this file changed)
----------------------------------------
AngleEmbedding treats its inputs as rotation angles in RADIANS. With a bare
Linear+tanh projection the outputs had std ~0.09, i.e. rotations of ~0.09 rad.
Since <Z> = cos(theta) for an Rx rotation from |0>, every input mapped to
<Z> ~ 0.996 and the circuit emitted a near-constant vector. The classifier head
then had nothing to separate and training loss sat at ln(3) ~ 1.0986.

Unparameterised CNOT rings compound the collapse at each layer, which is why
L=1 trained but L>=2 did not, and why small weight initialisation alone did not
help: the weights were never the problem, the input angle range was.

Two corrections:
  1. BatchNorm1d after the linear projection, so the pre-tanh signal has unit
     variance and tanh spans a useful part of its range.
  2. A learnable angle_scale (init pi/2) multiplying the tanh output, so
     encoded rotations cover a meaningful arc rather than a sliver near zero.

BatchNorm is applied in the CLASSICAL arm too. It is part of the shared
bottleneck, not part of the quantum mechanism, so it must be present in both or
the control is no longer matched. angle_scale is quantum-only, like the circuit
weights, because a classical linear head has no angular parameterisation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# quantum-inspired layer
# ---------------------------------------------------------------------------

def build_quantum_layer(n_qubits: int, n_layers: int,
                        init_std: float = 0.1) -> nn.Module:
    """AngleEmbedding -> BasicEntanglerLayers x L -> Pauli-Z expectations."""
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    init_method = {
        "weights": lambda t: torch.nn.init.normal_(t, mean=0.0, std=init_std)
    }
    return qml.qnn.TorchLayer(
        circuit, {"weights": (n_layers, n_qubits)}, init_method=init_method)


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
            self.angle_scale = None
            head_in = feat_dim
        else:
            # shared bottleneck: identical in classical and quantum arms
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, cfg.n_qubits),
                nn.BatchNorm1d(cfg.n_qubits),
                nn.Tanh(),
            )
            if self.mode == "quantum":
                self.quantum = build_quantum_layer(cfg.n_qubits, cfg.n_layers)
                # learnable; grouped with circuit params in the optimiser
                self.angle_scale = nn.Parameter(torch.tensor(math.pi / 2))
            else:
                self.quantum = None
                self.angle_scale = None
            head_in = cfg.n_qubits

        self.head = nn.Linear(head_in, n_classes)

    def quantum_parameters(self):
        """Circuit angles + angle_scale. These are parameterised on a radian
        scale and need their own learning rate; train.py groups them."""
        if self.quantum is None:
            return []
        ps = list(self.quantum.parameters())
        if self.angle_scale is not None:
            ps.append(self.angle_scale)
        return ps

    def to(self, *args, **kwargs):
        """Move as usual, then pull the quantum layer back to CPU.

        default.qubit allocates its state vector on CPU; leaving circuit
        weights on CUDA raises 'Expected all tensors to be on the same device'.
        """
        out = super().to(*args, **kwargs)
        if out.quantum is not None:
            out.quantum.to("cpu")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.proj(z)
        if self.quantum is not None:
            dev = z.device
            angles = (z * self.angle_scale).to("cpu")
            z = self.quantum(angles).float().to(dev)
        return self.head(z)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    q = sum(p.numel() for p in model.quantum_parameters()) \
        if hasattr(model, "quantum_parameters") else 0
    return {"total": total, "quantum": q, "classical": total - q}