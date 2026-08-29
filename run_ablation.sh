#!/usr/bin/env bash
# Full ablation grid. Run after a single configuration has trained cleanly.
#
# The arms that matter are the classical-bottleneck controls. If the quantum
# arms do not beat them at matched width, the central claim does not hold and
# you report that.

set -e

echo "=== Arm A: backbone only ==="
python train.py --bottleneck none

for N in 4 6; do
  echo "=== Arm B: classical bottleneck, width $N ==="
  python train.py --bottleneck classical --n_qubits "$N"

  for L in 1 2 3; do
    echo "=== Arm C/D: quantum, $N qubits, $L layers ==="
    python train.py --bottleneck quantum --n_qubits "$N" --n_layers "$L"
  done
done

echo "All runs complete. Results in ./runs/*/results.json"
