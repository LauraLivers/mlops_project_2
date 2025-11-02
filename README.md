**TBD**


Docker commands - do a training run immediately
```shell
# Build the image first (only when Dockerfile changes)
docker compose build

# Run the image an execute training run - no flags mean best run from project 1 (default values)
docker compose run hp-tuning --flag [flag_value]

# Destroy run (?)
docker compose down
```

checkpoints are not part of the version control (too big) - don't forget to save them separately
logs could be saved, change .gitignore


TensorBoard command
```shell
tensorboard --logdir logs/hp_tuning --port 6006 --host localhost --reload_interval=5
```
TensorBoard can only do HTTP - if browser doesn't allow: type manually 'https://localhost:6006' instead of clicking the link in the terminal





