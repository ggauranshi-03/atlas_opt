#!/usr/bin/env bash
# Train every optimizer in a config, then run the PTQ benchmark on its final checkpoint.
# Usage: bash run_ptq_campaign.sh <config.yaml> <epochs> <gpu_id> [comma,separated,optimizer,subset]
# Resumable: optimizers whose final checkpoint exists are not retrained; PTQ is skipped if its JSON exists.
# When an optimizer subset is given, the lock is scoped to that subset so disjoint subsets of the
# same config can run in parallel on different GPUs without colliding.
set -u
CONFIG=$1
EPOCHS=$2
GPU=$3
SUBSET=${4:-}
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=$GPU

LOCK_TAG=$(basename "$CONFIG" .yaml)${SUBSET:+_$(echo "$SUBSET" | tr ',' '_')}
LOCK=/tmp/ptq_campaign_${LOCK_TAG}.lock
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] another campaign for $CONFIG (subset: ${SUBSET:-all}) is already running (lock: $LOCK) - exiting"
    exit 1
fi
trap 'flock -u 9' EXIT

CONFIG_ID=$(python -c "import yaml,os;c=yaml.safe_load(open('$CONFIG'));print(c.get('experiment',{}).get('experiment_id',os.path.splitext(os.path.basename('$CONFIG'))[0]))")
if [ -n "$SUBSET" ]; then
    OPTS=$(echo "$SUBSET" | tr ',' ' ')
else
    OPTS=$(python -c "import yaml;print(' '.join(yaml.safe_load(open('$CONFIG'))['optimizers'].keys()))")
fi
LOGDIR=logs/ptq_campaign
mkdir -p "$LOGDIR"
echo "[$(date '+%F %T')] campaign $CONFIG_ID on GPU $GPU | epochs=$EPOCHS | optimizers: $OPTS"

for OPT in $OPTS; do
    CKPT=checkpoints/${CONFIG_ID}_${OPT}_epoch${EPOCHS}.pt
    for ATTEMPT in 1 2; do
        [ -f "$CKPT" ] && break
        echo "[$(date '+%F %T')] train $OPT (attempt $ATTEMPT)"
        python -u main_experiment.py -c "$CONFIG" -o "$OPT" -e "$EPOCHS" > "$LOGDIR/${CONFIG_ID}_${OPT}_train.log" 2>&1
    done
    if [ ! -f "$CKPT" ]; then
        echo "[$(date '+%F %T')] FAILED training $OPT (no $CKPT) - see $LOGDIR/${CONFIG_ID}_${OPT}_train.log"
        continue
    fi
    if [ -f "ptq_results/${CONFIG_ID}/${OPT}.json" ]; then
        echo "[$(date '+%F %T')] PTQ for $OPT already done"
        continue
    fi
    echo "[$(date '+%F %T')] PTQ $OPT (AWQ only, INT4 only)"
    python -u run_ptq.py -c "$CONFIG" -o "$OPT" -ckpt "$CKPT" --methods awq --settings 4:128 > "$LOGDIR/${CONFIG_ID}_${OPT}_ptq.log" 2>&1 \
        || echo "[$(date '+%F %T')] FAILED PTQ $OPT - see $LOGDIR/${CONFIG_ID}_${OPT}_ptq.log"
done
echo "[$(date '+%F %T')] campaign $CONFIG_ID finished"
