# Running training on TRUBA (ARF cluster)

This is the TRUBA equivalent of `notebooks/colab_train.ipynb`. `scripts/train_baseline.py`
itself needs no changes — it's a plain argparse script with no Colab-specific code. What
changes is how the job gets to a GPU: TRUBA is a shared SLURM cluster, so training runs as
a submitted batch job instead of a notebook cell.

**Never run training or anything GPU-bound directly on the login node** (`arf-ui1`) — it's
shared with every other user on the cluster and has no GPU attached anyway. Everything GPU
or long-running goes through `sbatch`/`salloc`. Lightweight, CPU-only checks (like
`scripts/check_pipeline.py`) are fine to run directly on the login node.

## 1. One-time environment setup

Run on the login node (no GPU needed, just internet access, which the login node has):

```bash
module load apps/truba-ai/gpu-2024.0   # miniforge/conda for GPU partitions
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
for flower (same numbers as the Colab notebook's Section 5).

## 3. Submit a training job

```bash
sbatch scripts/train_truba.sbatch configs/mushroom.yaml 15
sbatch scripts/train_truba.sbatch configs/flower.yaml 15
```

Check status with `squeue -u $USER`. Output/errors land in `logs/`. Checkpoints go to
`outputs/checkpoints/` under `/arf`, so they persist regardless of what happens to the job.
Training metrics stream live to Weights & Biases (see section 6) as the job runs.

GPU partitions on this cluster: `akya-cuda` (4 GPUs/node) and `barbun-cuda` (2 GPUs/node),
both with a 3-day walltime limit. The submitted script requests 1 GPU and 1 day; adjust
`--time` / `--partition` in `scripts/train_truba.sbatch` if a run needs longer or the other
partition is less busy (`sinfo -p akya-cuda` / `sinfo -p barbun-cuda`).

## 4. Resuming after a walltime cutoff or interruption

A checkpoint is saved after every epoch, so a killed/expired job can continue from where it
left off:

```bash
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
referencing `docs/training_run_log.md`.

One-time setup (per machine/account), so `sbatch` jobs authenticate automatically without
needing a key hardcoded anywhere:

```bash
module load apps/truba-ai/cpu-2024.0   # or gpu-2024.0 on a GPU partition
conda activate visionlab
wandb login   # paste your API key from wandb.ai/authorize — stored in ~/.netrc
```

To disable W&B for a one-off run (e.g. a smoke test), pass `--no-wandb` to
`scripts/train_baseline.py`.
