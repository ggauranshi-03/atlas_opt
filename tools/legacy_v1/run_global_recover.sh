#!/usr/bin/env bash
# Recovery after the disk filled at 13:47: retrain nanoGPT muon_sam_gfrob (GPU 7), forgetting sweeps for the finished runs, then the rest.
set -u
cd "$(dirname "$0")/.."
exec >> forgetting_results/logs/global_pipeline.log 2>&1
stamp() { echo "[$(date '+%F %T')] $*"; }
NEW="muon_sam_gfrob,fsam_gfrob_muon_stale,fsam_gfrob_muon_stale_momentum"
stamp "recovery: retrain nanoGPT muon_sam_gfrob on GPU 7"
timeout 10h python3 tools/experiments.py work --queue queues/global.json --gpu 7 > logs/queue/global_worker_g7_retry.log 2>&1 &
W=$!
stamp "recovery: forgetting sweeps on GPUs 0-4,6"
python3 tools/forgetting.py launch --gpus 0,1,2,3,4,6 --optimizers "$NEW"
wait $W
stamp "recovery: remaining sweeps on all GPUs"
python3 tools/forgetting.py launch --gpus 0,1,2,3,4,6,7 --optimizers "$NEW"
stamp "recovery: analysis"
python3 tools/forgetting.py analyze
stamp "pipeline finished"
