#!/bin/bash
set -e

echo "[Neural ODE] Testing integration..."
python -m physinet.train_ode \
  --data_dir data/marine_wave_dataset \
  --save_path checkpoints/ode_fno.pt
