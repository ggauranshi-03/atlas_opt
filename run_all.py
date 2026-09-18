#!/home/anant/Bhaskera/.venv/bin/python
import os
import sys
import subprocess

configs = [
    "configs/paper1_exp1_nanogpt_fineweb.yaml",
    "configs/paper1_exp2_nanogpt_ablations.yaml",
    "configs/paper1_exp3_cifar10_cnn.yaml",
    "configs/paper1_exp4_cifar10_batch_scaling.yaml",
    "configs/paper2_exp1_pythia70m_schatten_sweep.yaml",
    "configs/paper2_exp2_pythia70m_noise_analysis.yaml",
    "configs/paper2_exp3_pythia70m_pretrain_chinchilla.yaml",
]

python_bin = "/home/anant/Bhaskera/.venv/bin/python"
os.makedirs("logs", exist_ok=True)
summary_file = "logs/all_experiments_summary.csv"
if os.path.exists(summary_file):
    os.remove(summary_file)

master_log = open("logs/run_all.log", "w", buffering=1)

epochs = "5"
for idx, arg in enumerate(sys.argv):
    if arg in ("--epochs", "-e") and idx + 1 < len(sys.argv):
        epochs = sys.argv[idx + 1]

print(f"========================================================================")
print(f"          RUNNING ALL 7 EXPERIMENTS FOR {epochs} EPOCHS")
print(f"========================================================================")

for cfg in configs:
    cfg_id = os.path.splitext(os.path.basename(cfg))[0]
    log_file = f"logs/{cfg_id}.log"
    msg = f"\n========================================================================\n>>> Executing: {cfg} [Log: {log_file}] <<<\n========================================================================\n"
    print(msg, flush=True)
    master_log.write(msg)
    master_log.flush()

    with open(log_file, "w", buffering=1) as out:
        cmd = [python_bin, "-u", "exp1_atlas_baseline.py", "--config", cfg, "--optimizer", "all", "--epochs", epochs]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            out.write(line)
            out.flush()
            master_log.write(line)
            master_log.flush()
        proc.wait()

master_log.close()
print("\n========================================================================")
print("          ALL EXPERIMENTS FINISHED! LOGS SAVED IN logs/")
print("========================================================================")
