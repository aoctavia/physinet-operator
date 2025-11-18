#!/bin/bash
set -e

echo "[Diffusion model] Training..."
python -m physinet.train_diffusion \
  --data_dir data/marine_wave_dataset \
  --epochs 10 \
  --batch_size 4 \
  --lr 1e-4 \
  --timesteps 1000 \
  --save_path checkpoints/diffusion_wave.pt
