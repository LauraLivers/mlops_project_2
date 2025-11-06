# MLOPS Training Pipeline with Docker

## Overview
This project implements a PyTorch Lightning training pipeline for fine-tuning DistilBERT on the GLUE MRPC task. The pipeline supports hyperparameter tuning, automatic checkpointing, and TensorBoard logging. 

### What the Code Does
- Fine-tunes `distilber-base-uncased` on the MRPC (Microsoft Research Paraphrase Corpus) dataset
- Supports multiple optimizers (Adam, AdamW, SGD) and learning rate scheduler (linear, cosine, constant)
-  Automatically saves checkpoints and training logs
- Supports resuming training from checkpoints
- Provides real-time metrics via TensorBoard

## Project Structure
```
├── dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── requirements-base.txt       # Python dependencies
├── requirements-cuda.txt       # PyTorch for CUDA/MPS 
├── requirements-rocm.txt       # PyTorch for AMD
├── requirements-cpu.txt        # PyTorch for CPU
├── mlops_hp_2.py               # Main training script
├── checkpoints/                # Model checkpoints (created at runtime, not in git)
│   └── run_XXX_[params]/
│       └── last.ckpt
├── logs/                       # TensorBoard logs (created at runtime)
│   └── hp_tuning/
│       └── run_XXX_[params]/
│           └── events.out.tfevents.*
├── README.md
├── .gitignore                  # adapt to own environment
└── shell.sh                    # colour coding for terminal output. Run if convenient
```
**Note:** the logs and checkpoints folder will be created automatically during the first run. 

## Prerequisits
- Docker Engine and Docker Compose installed and running
- 8GB + RAM allocated to Docker (Settings $\rightarrow$ Resources)

## Quick Start

### 1. Build the Docker Image
Only needed once, or when Dockerfile/requirements change:
```shell
docker compose build --no-cache # use no-cache flag to build from scratch
```

### 2. Run Training
**With default hyperparameters**  
Run the command containing the hardware of the local machine. 
```shell
docker compose run hp-tuning-amd --flag [flag_value] # AMD
# or
docker compose run hp-tuning-cuda --flag [flag_value] # NVIDIA
# or
docker compose run hp-tuning --flag [flag_value] # for CPU
```
Note, that the AMD-build has only been tested on AMD Ryzen 7 PRO 8840U. If any other AMD chip is used, changes might have to be made to `docker-compose.yml` and `requirements-amd.txt`.

**Available flags**
- `--learning_rate` (default: 2e-5)
- `--epochs` (default: 3)
- `--train_batch_size` (default: 32)
- `--eval_batch_size` (default: 32)
- `--optimizer_type` (adam, adamw, sgd)
- `--lr_scheduler` (linear, cosine, constant)
- `--warmup_steps` (default: 0)
- `--weight_decay` (default: 0.0)
- `--adam_beta1`  (default: 0.9)
- `--adam_beta2`  (default: 0.999)
- `--fresh_start` (True/False - ignore existing checkpoints if `True`)

### 3. Clean up
Stop and remove containers:
```shell
docker compose down
```

## Monitoring with TensorBoard
### Start TensorBoard
run this from inside the root folder:
```shell
tensorboard --logdir logs/hp_tuning --port 6006 --host localhost --reload_interval 5
```

### Access TensorBoard
TensorBoard only serves HTTP. If your browser blocks HTTP on localhost:
- **type manually:** `http://127.0.0.1:6006` (instead of clicking the terminal link)
- **Or use:** `https://localhost:6006`

## Training Details

### Checkpointing
- Checkpoints saved to `./checkpoints/run_[run_number]_[hyperparameters]/last.ckpt`
- Training automatically resumes from the last checkpoint unless `--fresh_start True`
- **Important:** Checkpoints are NOT in git - save them separately for important runs

### Logs
- TensorBoard logs saved to `./logs/hp_tuning/run_[run_number]_[hyperparameters]/`
- Logs can be committed to git (update `.gitignore` if needed)
- Each run has a unique name based on order and hyperparameters

### Performance
- **Local:** with GPU support ~3 minutes for 3 Epochs
- **Docker:** with GPU support ~3 minutes for 3 Epochs, with CPU ~20 minutes for 3 Epochs

Note, that on a Mac computer there is no GPU support inside the Docker. In this case, run the code native on the local machine:
```shell
python3 -m venv .venv # new virtual environment
source .venv/bin/activate 
pip3 install requirements-base.txt requirements-cuda.txt # CUDA and MPS use the same PyTorch version
python3 mlops_hp_2.py --flag [flag_value]
```
## Troubleshooting
### Docker build takes too longs
The first build downloads ~2GB of dependencies. Subsequent builds use cache and should be significantly faster.

### Training is very slow
If GPU is available on the local machine, check logs to ensure the hardware is correctly detected during the build process.

### TensorBoard shows blank page
- ensure you've run at least one training session
- Check that `logs/hp_tuning/`contains event files
- Try hard refresh (Cmd+Shift+R)
- Ensure you're running the command from the root directory

### Can't delete Docker image
```shell
docker rm $(docker ps -a -q --filter ancestor=IMAGE_NAME)
docker rmi IMAGE_NAME
```


