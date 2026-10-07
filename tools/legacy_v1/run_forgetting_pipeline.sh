#!/usr/bin/env bash
# Unattended E1 pipeline for sgd, adam, muon_sam, fsam_muon, fsam (runs detached; safe to close the session).
#   1. forgetting sweeps for sgd/adam (existing checkpoints) on GPU 7, while the rho=0.01 retrains finish
#   2. wait until the six retrained base checkpoints exist
#   3. every remaining sweep on all free GPUs
#   4. re-run the 9-optimizer analysis and log it to the Forgetting-* W&B projects
# Completed sweeps are skipped, so the script can simply be restarted if interrupted.
#   setsid nohup bash tools/run_forgetting_pipeline.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p forgetting_results/logs
exec >> forgetting_results/logs/pipeline.log 2>&1
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "phase 1: sgd/adam sweeps on GPU 7"
python3 tools/forgetting.py launch --gpus 7 --optimizers sgd,adam

stamp "phase 2: waiting for the rho=0.01 retrained checkpoints"
until python3 - <<'EOF'
import os, sys, time
ckpts = [f"checkpoints/{cid}_{opt}_seed42_epoch{ep}.pt"
         for cid, ep in (("nanogpt_fineweb", 15), ("pythia70m_pretrain_chinchilla", 14))
         for opt in ("muon_sam", "fsam_muon", "fsam")]
ready = all(os.path.exists(c) and time.time() - os.path.getmtime(c) > 120 for c in ckpts)  # fully written
sys.exit(0 if ready else 1)
EOF
do sleep 60; done
stamp "all retrained checkpoints present"

stamp "phase 3: remaining sweeps on all GPUs"
python3 tools/forgetting.py launch --gpus 0,1,2,3,4,6,7 --optimizers sgd,adam,muon_sam,fsam_muon,fsam

stamp "phase 4: analysis"
python3 tools/forgetting.py analyze
stamp "pipeline finished"
