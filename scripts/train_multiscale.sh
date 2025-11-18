#!/bin/bash
set -e

echo "[Multiscale FNO] Training..."
python -m physinet.train_multiscale \
  --data_dir data/marine_wave_dataset \
  --epochs 10 \
  --batch_size 4 \
  --lr 1e-3 \
  --lambda_phys 1.0 \
  --dt 0.1 \
  --c 1.0 \
  --save_path checkpoints/ms_fno.pt
