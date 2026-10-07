#!/usr/bin/env bash
# Unattended pipeline for the spectral/Frobenius pair comparison (runs detached; safe to close the session).
#   1. train, at the current rho=0.01 configs, the six optimizers that lack consistent checkpoints
#      (3 new + 3 partners trained earlier at rho=0.1) on nanoGPT and Pythia, one job per GPU from a shared queue
#      (each job also runs its INT4 PTQ step)
#   2. forgetting sweeps for those six on all GPUs
#   3. 15-optimizer analysis, logged to the Forgetting-* W&B projects
# Finished work is skipped, so the script can simply be restarted if interrupted.
#   setsid nohup bash tools/run_pairs_pipeline.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p forgetting_results/logs logs/queue
exec >> forgetting_results/logs/pairs_pipeline.log 2>&1
stamp() { echo "[$(date '+%F %T')] $*"; }

GPUS="0 1 2 3 4 6 7"
GPU_CSV="0,1,2,3,4,6,7"
# double-pass methods first (longest), single-pass stale methods last
NEW="sam_ortho,fsam_ortho,fsam_ortho_muon,muon_sam_frob,fsam_frob_muon_stale,fsam_frob_muon_stale_momentum"
Q=queues/pairs.json

stamp "phase 1: queue training jobs"
python3 tools/experiments.py final --config configs/nanogpt_fineweb.yaml --epochs 15 --seeds 42 --optimizers "$NEW" --queue $Q
python3 tools/experiments.py final --config configs/pythia70m_pretrain_chinchilla.yaml --epochs 14 --seeds 42 --optimizers "$NEW" --queue $Q

pids=()
for g in $GPUS; do
  timeout 10h python3 tools/experiments.py work --queue $Q --gpu $g > logs/queue/pairs_worker_g$g.log 2>&1 &
  pids+=($!)
done
stamp "workers started on GPUs $GPUS"
wait "${pids[@]}"
stamp "training workers finished"

python3 - <<'EOF'
import json, os, sys
missing = [j["id"] for j in json.load(open("queues/pairs.json")) if j["status"] != "done"]
ck = [f"checkpoints/{c}_{o}_seed42_epoch{e}.pt" for c, e in (("nanogpt_fineweb", 15), ("pythia70m_pretrain_chinchilla", 14))
      for o in "sam_ortho,fsam_ortho,fsam_ortho_muon,muon_sam_frob,fsam_frob_muon_stale,fsam_frob_muon_stale_momentum".split(",")]
absent = [c for c in ck if not os.path.exists(c)]
print("jobs not done:", missing or "none", "| checkpoints absent:", absent or "none")
EOF

stamp "phase 2: forgetting sweeps on all GPUs"
python3 tools/forgetting.py launch --gpus $GPU_CSV --optimizers "$NEW"

stamp "phase 3: analysis"
python3 tools/forgetting.py analyze
stamp "pipeline finished"
