#!/usr/bin/env bash
# SOMA-PreNS5 and OP-SOMA-PostNS5 under the v2 protocol (same as every other optimizer):
#   rho tuning (seed 0; lr/momentum/wd/aux-AdamW lr inherited from the tuned Muon) -> 3 seeds x 3 models + INT4 PTQ
#   -> forgetting sweeps (2 LMs x 3 seeds x 4 datasets) -> analysis + summaries 
#   setsid nohup bash run_soma_projected.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
exec >> logs/pipeline_soma_projected.log 2>&1
stamp() { echo "[$(date '+%F %T')] $*"; }
GPUS=0,1,2,3,4,6,7
OPTS=soma_prens5,op_soma_postns5
step() { stamp "START $1"; shift; "$@" || { stamp "FAILED: $*"; exit 1; }; }

step "rho tuning"   python3 -u tools/tune.py rho   --gpus $GPUS --optimizers $OPTS --tag _somaproj
stamp "disk before final runs: $(df -h . | tail -1)"
step "final runs + INT4 PTQ"   python3 -u tools/tune.py final --gpus $GPUS --optimizers $OPTS --tag _somaproj
python3 tools/experiments.py status --queue queues/final_somaproj.json
stamp "waiting for the earlier recovery script (finish_v2.sh) before the forgetting stage"
while pgrep -f "bash finish_v2.sh" > /dev/null; do sleep 60; done
step "forgetting sweeps"       python3 -u tools/forgetting.py launch --gpus $GPUS --optimizers $OPTS
step "retry missing sweeps"    python3 -u tools/forgetting.py launch --gpus $GPUS --optimizers $OPTS
step "forgetting analysis"     python3 -u tools/forgetting.py analyze
step "summaries"               python3 -u tools/summarize.py

echo "[$(date '+%F %T')] ALL DONE"
