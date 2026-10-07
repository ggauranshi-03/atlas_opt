#!/usr/bin/env bash
# v2 pipeline, unattended: tuning (lr, then rho) -> 9 optimizers x 3 models x 3 seeds (+ INT4 PTQ) -> forgetting -> summaries.
# Every stage is resumable: re-running this script skips finished work.
#   setsid nohup bash run_v2.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs/queue results
exec >> logs/pipeline.log 2>&1
stamp() { echo "[$(date '+%F %T')] $*"; }
GPUS=0,1,2,3,4,6,7          # GPU 5 is fully used by another user; 3, 4, 6 are shared with other users
step() { stamp "START $1"; shift; "$@" || { stamp "FAILED: $*"; exit 1; }; }

step "stage 1: learning-rate tuning"   python3 -u tools/tune.py lr  --gpus $GPUS
step "stage 2: rho tuning"             python3 -u tools/tune.py rho --gpus $GPUS
step "stage 3: final runs + INT4 PTQ"  python3 -u tools/tune.py final --gpus $GPUS
stamp "disk after training: $(df -h . | tail -1)"
step "stage 4: forgetting sweeps"      python3 -u tools/forgetting.py launch --gpus $GPUS
step "stage 4b: retry missing sweeps"  python3 -u tools/forgetting.py launch --gpus $GPUS
step "stage 5: forgetting analysis"    python3 -u tools/forgetting.py analyze
step "stage 6: summaries"              python3 -u tools/summarize.py
stamp "ALL DONE"
