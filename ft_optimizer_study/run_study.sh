#!/usr/bin/env bash
# Unattended pipeline: link base checkpoints -> calibrate learning-rate grids -> all fine-tuning sweeps -> tables. Every stage is resumable (finished jobs are skipped).
#   setsid nohup bash ft_optimizer_study/run_study.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p ft_optimizer_study/logs
exec >> ft_optimizer_study/logs/pipeline.log 2>&1
GPUS=0,1,2,3,4,6,7                     # GPU 5 belongs to another user
S="python3 -u ft_optimizer_study/ft_study.py"
stamp() { echo "[$(date '+%F %T')] $*"; }
step() { stamp "START $1"; shift; "$@" || { stamp "FAILED: $*"; exit 1; }; }
stamp "disk: $(df -h . | tail -1)"
step "setup"                               $S setup
step "calibration of learning-rate grids"  $S calibrate --gpus $GPUS
step "fine-tuning sweeps (2 models x 3 bases x 3 seeds x 11 optimizers)" $S launch --gpus $GPUS
step "retry jobs without a result"         $S launch --gpus $GPUS
step "tables"                              $S analyze
stamp "ALL DONE"
