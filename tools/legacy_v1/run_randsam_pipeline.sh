#!/usr/bin/env bash
# RandSAM-Muon (random-direction global-Frobenius SAM + Muon): train (+INT4 PTQ) on both LMs, forgetting sweeps, analysis.
# Disk is nearly full, so while training only the newest epoch checkpoint of these two runs is kept.
set -u
cd "$(dirname "$0")/.."
mkdir -p forgetting_results/logs logs/queue
exec >> forgetting_results/logs/randsam_pipeline.log 2>&1
stamp() { echo "[$(date '+%F %T')] $*"; }
Q=queues/randsam.json
stamp "queue"
python3 tools/experiments.py final --config configs/nanogpt_fineweb.yaml --epochs 15 --seeds 42 --optimizers randsam_muon --queue $Q
python3 tools/experiments.py final --config configs/pythia70m_pretrain_chinchilla.yaml --epochs 14 --seeds 42 --optimizers randsam_muon --queue $Q

( while true; do
    for c in nanogpt_fineweb pythia70m_pretrain_chinchilla; do
      latest=$(ls checkpoints/${c}_randsam_muon_seed42_epoch*.pt 2>/dev/null | sed 's/.*epoch\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
      [ -n "$latest" ] && for f in checkpoints/${c}_randsam_muon_seed42_epoch*.pt; do
        e=${f##*epoch}; e=${e%.pt}; [ "$e" -lt $((latest-1)) ] && rm -f "$f"
      done
    done
    sleep 60
  done ) &
PRUNER=$!

pids=()
for g in 0 1; do
  timeout 10h python3 tools/experiments.py work --queue $Q --gpu $g > logs/queue/randsam_worker_g$g.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
kill $PRUNER 2>/dev/null
for c in nanogpt_fineweb pythia70m_pretrain_chinchilla; do
  latest=$(ls checkpoints/${c}_randsam_muon_seed42_epoch*.pt 2>/dev/null | sed 's/.*epoch\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
  for f in checkpoints/${c}_randsam_muon_seed42_epoch*.pt; do e=${f##*epoch}; e=${e%.pt}; [ "$e" != "$latest" ] && rm -f "$f"; done
done
stamp "training finished; final checkpoints: $(ls checkpoints/*randsam_muon*)"
python3 tools/forgetting.py launch --gpus 0,1,2,3,4,6 --optimizers randsam_muon
stamp "waiting for the earlier recovery pipeline (GPU 7 / analysis) before the final analysis"
while pgrep -f run_global_recover.sh > /dev/null; do sleep 60; done
python3 tools/forgetting.py launch --gpus 0,1,2,3,4,6,7 --optimizers randsam_muon,muon_sam_gfrob
python3 tools/forgetting.py analyze
python3 tools/make_tikz.py
stamp "pipeline finished"
