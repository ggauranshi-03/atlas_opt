#!/usr/bin/env bash
# CIFAR-10 remainder: train+PTQ RandSAM-Muon, and PTQ (checkpoints exist) for adam, sgd, fsam, muon_sam, fsam_ortho_muon.
set -u
cd "$(dirname "$0")/.."
exec >> logs/queue/cifar_remaining.log 2>&1
stamp() { echo "[$(date '+%F %T')] $*"; }
Q=queues/cifar.json
python3 tools/experiments.py final --config configs/cifar10_cnn.yaml --epochs 7 --seeds 42 --optimizers randsam_muon --queue $Q
pids=()
for g in 0 1 2 3 4 7; do
  timeout 5h python3 tools/experiments.py work --queue $Q --gpu $g > logs/queue/cifar_remaining_g$g.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
python3 tools/experiments.py status --queue $Q
stamp "ALL DONE"
