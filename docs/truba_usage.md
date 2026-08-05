# Running training on TRUBA (ARF cluster)

`scripts/train_baseline.py` is a plain argparse script — the same one whether it's triggered
locally or on TRUBA, no code changes needed. What TRUBA adds is that it's a shared SLURM
cluster: training runs as a submitted batch job instead of an interactive process, so someone
else's runs don't get starved and a dropped SSH connection doesn't kill an in-progress job.
The project's actual runs so far are CPU-only on the `orfoz` partition (see
`scripts/train_truba_cpu.sbatch`) — a GPU entry point also exists
(`scripts/train_truba.sbatch`) for when that's worth using instead.

**Never run training or anything long-running directly on the login node** (`arf-ui1`) — it's
shared with every other user on the cluster and has no GPU attached anyway. Everything
GPU-bound or long-running goes through `sbatch`/`salloc`. Lightweight, CPU-only checks (like
`scripts/check_pipeline.py`) are fine to run directly on the login node.

## 1. One-time environment setup

Run on the login node (no GPU needed, just internet access, which the login node has):

```bash
module load apps/truba-ai/cpu-2024.0   # miniforge/conda; same env works from GPU partitions too
conda create -n visionlab python=3.10 -y
conda activate visionlab
cd /arf/scratch/hasozkan/sum/VisionLab
pip install -e .
```

## 2. Download the full datasets

`/arf` is a persistent, shared Lustre filesystem — unlike Colab's local disk, nothing here
gets wiped when a session ends, so there's no Drive-mount equivalent needed.

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username": "YOUR_KAGGLE_USERNAME", "key": "YOUR_KAGGLE_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

pip install -q kaggle
mkdir -p datasets/extracted/mushroom
kaggle datasets download -d zlatan599/mushroom1 -p datasets/extracted/mushroom --unzip
kaggle datasets download -d waseemalastal/the-oxford-flowers-102-dataset -p datasets/extracted --unzip
```

Verify the download landed correctly:

```bash
python scripts/check_pipeline.py
```

Expect train/val/test counts of 689,520 / 15,616 / 15,614 for mushroom and 6,552 / 818 / 819
for flower.

## 3. Submit a training job

Before a long run, `scripts/smoke_test.sbatch` is worth running first — it's a 1-epoch,
32-image sanity check on the `debug` partition (queues fast, 4h cap) that catches data
loading / logging / checkpoint bugs before committing a multi-day `orfoz` allocation:

```bash
sbatch scripts/smoke_test.sbatch configs/mushroom.yaml
```

Default (CPU, `orfoz` — what this project actually trains on):

```bash
sbatch scripts/train_truba_cpu.sbatch configs/mushroom.yaml 15
sbatch scripts/train_truba_cpu.sbatch configs/flower.yaml 15
```

GPU alternative, if a GPU partition is worth using for a given run:

```bash
sbatch scripts/train_truba.sbatch configs/mushroom.yaml 15
sbatch scripts/train_truba.sbatch configs/flower.yaml 15
```

Check status with `squeue -u $USER`. Output/errors land in `logs/`. Checkpoints go to
`outputs/checkpoints/` under `/arf`, so they persist regardless of what happens to the job.
Training metrics stream live to Weights & Biases (see section 6) as the job runs.

`orfoz` has a 3-day walltime limit and `train_truba_cpu.sbatch` requests the whole node
(`--exclusive`, 56 cores) — node-sharing was measured to roughly double epoch time. GPU
partitions: `akya-cuda` (4 GPUs/node) and `barbun-cuda` (2 GPUs/node), also 3-day walltime;
`train_truba.sbatch` requests 1 GPU and 1 day by default — adjust `--time`/`--partition` if a
run needs longer, or the other partition is less busy (`sinfo -p akya-cuda`/`-p barbun-cuda`).

## 4. Resuming after a walltime cutoff or interruption

A checkpoint is saved after every epoch, so a killed/expired job can continue from where it
left off — third argument is the checkpoint path, fourth (optional) caps training to a
stratified subset (N images per class):

```bash
sbatch scripts/train_truba_cpu.sbatch configs/mushroom.yaml 15 outputs/checkpoints/mushroom_resnet50.pt
sbatch scripts/train_truba.sbatch configs/mushroom.yaml 15 outputs/checkpoints/mushroom_resnet50.pt
```

## 5. Checking the GPU / a quick interactive session

For anything exploratory that needs a GPU (sanity-checking `torch.cuda.is_available()`,
debugging), open a short interactive allocation instead of using the login node:

```bash
salloc --partition=akya-cuda --gres=gpu:1 --time=00:10:00
# once granted:
nvidia-smi
exit   # release the allocation when done
```

## 6. Monitoring training

Training logs to [Weights & Biases](https://wandb.ai) (project `visionlab`) — no port
forwarding needed, just open the run's URL (printed to `logs/<job>.out` at startup, and in
your W&B dashboard) in any browser. Each run is named
`{dataset}_{model}_{subset_label}_ep{epochs}_{job_id}` so it's identifiable without cross
referencing `notebooks/07_training_run_log.ipynb`.

One-time setup (per machine/account), so `sbatch` jobs authenticate automatically without
needing a key hardcoded anywhere:

```bash
module load apps/truba-ai/cpu-2024.0   # or gpu-2024.0 on a GPU partition
conda activate visionlab
wandb login   # paste your API key from wandb.ai/authorize — stored in ~/.netrc
```

To disable W&B for a one-off run (e.g. a smoke test), pass `--no-wandb` to
`scripts/train_baseline.py`.

## 7. Automated flywheel retraining check (cron)

`scripts/check_flywheel_threshold.py` (see `RETRAINING_POLICY.md` decision 2) isn't triggered
by anything on its own — it's now wired into `crontab` (per-user, only affects `hasozkan`'s
own scheduled jobs) to run daily at 03:00/03:15 for mushroom/flower respectively:

```bash
crontab -l    # view the installed entries
crontab -e    # edit (opens $EDITOR)
```

It calls the `visionlab` conda env's python by absolute path
(`/arf/home/hasozkan/miniforge3/envs/visionlab/bin/python`) instead of `module load` +
`conda activate`, since cron's minimal environment doesn't source the shell init those need —
a bare `module`/`conda activate` in a crontab entry silently fails. Output (whether it
submitted `scripts/flywheel_retrain.sbatch` or found nothing to do) is appended to
`logs/flywheel_cron.log` — check that file if a scheduled run doesn't seem to have fired.
