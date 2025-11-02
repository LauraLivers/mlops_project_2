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
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── requirements.txt            # Python dependencies
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
- Docker Desktop installed and running
- 8GB + RAM allocated to Docker (Settings $\rightarrow$ Resources)

## Quick Start

### 1. Build the Docker Image
Only needed once, or when Dockerfile/requirements change:
```shell
docker compose build
```

### 2. Run Training
**With default hyperparameters**
```shell
docker compose run hp-tuning --flag [flag_value]
```

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
- `--fresh_start` (True/False - ignore existing checkpoints)

### 3. Clean up
Stop and remove containers:
```shell
docker compose down
```

## Monitoring with TensorBoard
### Start TensorBoard
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
- Each run has a unique name based on hyperparameters

### Performance
- **Local:** with MPS or CUDA support ~1 minute per run
- **Docker:** with CPU ~20 minutes for 3 Epochs

## Troubleshooting
### Docker build takes too longs
The first build downloads ~2GB of dependencies. Subsequent builds use cache and should be significantly faster

**If running the code in Github Codespaces:**
1. in `requirements.txt` comment out `torch==2.9.0` 
2. in `dockerfile` add `RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu` before the line installing the dependencies
This will install a lighter version of torch that should significantly reduce the build time and memory errors.

### Training is very slow
Docker on Mac can only use CPU. For faster training, run the script directly:
```shell
python3 mlops_hp_2.py --flag [flag_value]
```
The script is designed to use GPU/MPS if available.

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


