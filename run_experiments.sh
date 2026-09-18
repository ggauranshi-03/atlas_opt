#!/usr/bin/env bash
set -e

echo "========================================================================"
echo "          RUNNING ATLAS CPT EXPERIMENTS ON ALL CONFIGS"
echo "========================================================================"

PYTHON="/home/anant/Bhaskera/.venv/bin/python"
mkdir -p logs
rm -f logs/all_experiments_summary.csv

CONFIGS=(
    "configs/paper1_exp1_nanogpt_fineweb.yaml"
    "configs/paper1_exp2_nanogpt_ablations.yaml"
    "configs/paper1_exp3_cifar10_cnn.yaml"
    "configs/paper1_exp4_cifar10_batch_scaling.yaml"
    "configs/paper2_exp1_pythia70m_schatten_sweep.yaml"
    "configs/paper2_exp2_pythia70m_noise_analysis.yaml"
    "configs/paper2_exp3_pythia70m_pretrain_chinchilla.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    if [ -f "$cfg" ]; then
        cfg_name=$(basename "$cfg" .yaml)
        echo ""
        echo "========================================================================"
        echo ">>> Executing: $cfg [Log: logs/${cfg_name}.log] <<<"
        echo "========================================================================"
        $PYTHON -u exp1_atlas_baseline.py --config "$cfg" --optimizer all "$@" 2>&1 | tee "logs/${cfg_name}.log"
    fi
done

echo ""
echo "========================================================================"
echo "          ALL EXPERIMENTS FINISHED! LOGS SAVED IN logs/"
echo "========================================================================"
