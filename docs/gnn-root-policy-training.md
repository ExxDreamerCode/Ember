# GNN root policy training

This document records how `src/root_policy.onnx` was produced. The model is an
experimental root move-ordering policy. It does not replace Ember's NNUE
evaluator; it predicts a 64×73 root-move policy vector that can be used as an
additional ordering signal before iterative deepening starts.

The checked-in artifact was trained on 2026-07-31 on a Google Cloud VM:

- machine type: `g2-standard-4`
- accelerator: 1× NVIDIA L4
- boot disk: 250 GB
- OS: Ubuntu 22.04 LTS
- NVIDIA driver observed after setup: 595.84
- Python: 3.10 from the OS image
- PyTorch wheel: `torch==2.4.1+cu124`

Other CUDA-capable machines should also work, but the numbers below are the
reference configuration for rebuilding the current artifact.

## Files

The training pipeline lives under `training/gnn_root_policy/`:

- `extract_sfbinpack/` is a small Rust extractor for Stockfish binpack data.
- `train_root_policy.py` trains the Dense GCN and exports ONNX.
- `run_gcp_training.sh` installs dependencies, downloads data, extracts samples,
  trains, validates ONNX through `onnxruntime`, and writes an archive.

Generated archives, checkpoints, sample files, downloaded binpacks, and logs are
not tracked by Git.

## Data source

The current artifact uses this official Stockfish binpack:

```text
https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/test80-2022-08-aug-16tb7p.v6-dd.min.binpack?download=true
```

The extractor keeps positions with:

- ply at least 16;
- side to move not in check;
- absolute score at most 10000;
- quiet normal best moves only by default;
- a move that maps cleanly into the 64×73 AlphaZero-style policy layout.

The 2026-07-31 rebuild extracted 20,000,000 records:

```text
kept=20000000 seen=33978622 skipped_filter=13978622 skipped_map=0
```

Zero `skipped_map` is important. If this becomes nonzero, do not use the model
until the extractor/Rust policy-index mapping disagreement is understood.

## Creating a GCP training VM

If the project does not have GPU quota, request at least one `GPUS_ALL_REGIONS`
unit first. The 2026-07-31 run used one on-demand L4 VM and stayed far below
100 USD.

Create a VM similar to this:

```sh
PROJECT=your-project
ZONE=us-central1-a
NAME=ember-gnn-train

gcloud compute instances create "$NAME" \
  --project "$PROJECT" \
  --zone "$ZONE" \
  --machine-type g2-standard-4 \
  --accelerator type=nvidia-l4,count=1 \
  --maintenance-policy TERMINATE \
  --provisioning-model STANDARD \
  --boot-disk-size 250GB \
  --boot-disk-type pd-standard \
  --image-family ubuntu-2204-lts \
  --image-project ubuntu-os-cloud
```

Install and verify the NVIDIA driver:

```sh
gcloud compute ssh "$NAME" --zone "$ZONE"

sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common build-essential curl wget python3-venv
sudo apt-get install -y nvidia-driver-595-open
sudo reboot
```

After reconnecting:

```sh
nvidia-smi
```

The reference run reported an NVIDIA L4 with driver 595.84.

## Copying the training pipeline

From the repository root:

```sh
PROJECT=your-project
ZONE=us-central1-a
NAME=ember-gnn-train

gcloud compute ssh "$NAME" --zone "$ZONE" --command 'mkdir -p ~/ember-gnn/src'
gcloud compute scp --recurse training/gnn_root_policy/extract_sfbinpack \
  "$NAME:~/ember-gnn/src/extract_sfbinpack" --zone "$ZONE"
gcloud compute scp training/gnn_root_policy/train_root_policy.py \
  training/gnn_root_policy/run_gcp_training.sh \
  "$NAME:~/ember-gnn/src/" --zone "$ZONE"
```

## Training command

Run the same configuration used for the checked-in model:

```sh
gcloud compute ssh "$NAME" --zone "$ZONE" --command '
  set -euo pipefail
  cd ~/ember-gnn
  chmod +x src/run_gcp_training.sh
  SAMPLES=20000000 \
  EPOCHS=5 \
  BATCH_SIZE=8192 \
  HIDDEN=192 \
  LAYERS=4 \
  WORKERS=2 \
  VALID_BATCHES=128 \
  ./src/run_gcp_training.sh
'
```

The script creates:

```text
~/ember-gnn/artifacts/run-YYYYMMDD-HHMMSS/
~/ember-gnn/artifacts/run-YYYYMMDD-HHMMSS.tar.gz
~/ember-gnn/artifacts/latest-run.txt
```

For unattended runs, redirect output to a log and keep the PID:

```sh
SAMPLES=20000000 EPOCHS=5 BATCH_SIZE=8192 HIDDEN=192 LAYERS=4 \
  WORKERS=2 VALID_BATCHES=128 \
  nohup ./src/run_gcp_training.sh > logs/train-20m-h192-l4.log 2>&1 &
echo $! > logs/train-20m-h192-l4.pid
```

## Reference metrics

The checked-in artifact was produced by `run-20260731-224202`. Final validation:

```json
{
  "epoch": 5,
  "elapsed_s": 357.20570278167725,
  "samples_per_s": 53190.64016067159,
  "train_loss": 6.133073762663189,
  "train_samples": 19000000,
  "valid": {
    "loss": 6.09784094921875,
    "samples": 1000000,
    "top1": 0.021224,
    "top3": 0.050836,
    "top5": 0.076261
  }
}
```

The exported ONNX model was validated with `onnxruntime`:

```text
ok shape=(1, 4672) min=-196.77931213378906 max=219.55337524414062
```

The checked-in `src/root_policy.onnx` SHA-256 is:

```text
aeb50f4e9f2332030122a08a8daa078c0931cd87646807eb1e5ef3475dc92c52
```

## Downloading and installing the artifact

Download the archive:

```sh
RUN_DIR=$(gcloud compute ssh "$NAME" --zone "$ZONE" \
  --command 'cat ~/ember-gnn/artifacts/latest-run.txt')
ARCHIVE="$(basename "$RUN_DIR").tar.gz"

gcloud compute scp "$NAME:~/ember-gnn/artifacts/$ARCHIVE" \
  training/gnn_root_policy/artifacts/ --zone "$ZONE"
```

Extract and install the model:

```sh
mkdir -p training/gnn_root_policy/artifacts/extracted
tar -xzf "training/gnn_root_policy/artifacts/$ARCHIVE" \
  -C training/gnn_root_policy/artifacts/extracted

install -m 0644 \
  "training/gnn_root_policy/artifacts/extracted/$(basename "$RUN_DIR")/root_policy.onnx" \
  src/root_policy.onnx

sha256sum src/root_policy.onnx
```

Only commit `src/root_policy.onnx` after ONNX validation passes and the move-index
mapping is still shared by the extractor and Rust integration.

## Stopping the VM

Always stop the VM after downloading the result:

```sh
gcloud compute instances stop "$NAME" --zone "$ZONE" --quiet
gcloud compute instances describe "$NAME" --zone "$ZONE" --format='value(status)'
```

The status must become `TERMINATED`.
