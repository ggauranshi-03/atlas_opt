#!/usr/bin/env bash
# MSOMA (Momentum-SAM perturbation + Muon base, optimizers/msoma.py) under the v2 continued-pretraining protocol (same as every other optimizer):
#   rho tuning (seed 0; lr/momentum/wd/aux-AdamW lr inherited from the tuned Muon) -> 3 seeds x 3 models (INT4 PTQ is part of tools/tune.py final)
#   -> summaries. Not started yet; start with
#   setsid nohup bash run_msoma.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs
exec >> logs/pipeline_msoma.log 2>&1
stamp() { echo "[$(date '+%F %T')] $*"; }
GPUS=0,1,2,3,4,6,7
OPTS=msoma
step() { stamp "START $1"; shift; "$@" || { stamp "FAILED: $*"; exit 1; }; }

step "rho tuning"            python3 -u tools/tune.py rho   --gpus $GPUS --optimizers $OPTS --tag _msoma
stamp "disk before final runs: $(df -h . | tail -1)"
step "final runs (+ INT4 PTQ)"  python3 -u tools/tune.py final --gpus $GPUS --optimizers $OPTS --tag _msoma
python3 tools/experiments.py status --queue queues/final_msoma.json
step "summaries"            python3 -u tools/summarize.py

echo "[$(date '+%F %T')] ALL DONE"
