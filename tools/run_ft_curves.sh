#!/usr/bin/env bash
# Re-runs the fine-tuning recipe (one learning rate) for all 66 (model, optimizer, seed) groups and records held-out accuracy every 8 steps.
# Needs the final checkpoints in checkpoints/ and the fine-tuning data caches (python tools/forgetting.py prepare ...). 4 workers; edit GPUS.
#   bash tools/run_ft_curves.sh && python tools/make_ft_figures.py
cd "$(dirname "$0")/.."; mkdir -p paper_ft_curves/raw paper_ft_curves/logs
export GPUS="${GPUS:-0 1 2 3}"
python3 - << 'PY' > paper_ft_curves/logs/shards.txt
import os
opts = ["adam","sgd","muon","sam","fsam","muon_sam","fsam_ortho_muon","fsam_ortho_muon_stale_momentum","randsam_muon","soma_prens5","op_soma_postns5"]
groups = [f"{m}:{o}:{s}" for m in ("nanogpt","pythia70m") for s in (42,43,44) for o in opts]
W = len(os.environ.get("GPUS", "0 1 2 3").split())
for w in range(W):
    print(",".join(groups[w::W]))
PY
i=0; pids=()
for g in $GPUS; do
  i=$((i+1)); shard=$(sed -n "${i}p" paper_ft_curves/logs/shards.txt)
  CUDA_VISIBLE_DEVICES=$g python3 -u tools/ft_worker.py --groups "$shard" > paper_ft_curves/logs/worker_g$g.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
