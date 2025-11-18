# #!/bin/bash

# echo "Generating dataset..."
# python -m physigym.env --output_dir data/synthetic --num_trajectories 20 --pde_type wave2d

# echo "Training model..."
# python -m physinet.train --data_dir data/synthetic --epochs 5

# echo "Open notebook: notebooks/03_visualize_rollouts.ipynb"


#!/bin/bash
set -e

echo "Generating dataset (basic wave2d)..."
python -m physigym.env \
  --output_dir data/synthetic \
  --num_trajectories 20 \
  --pde_type wave2d

echo "Training baseline MarineFNO..."
python -m physinet.train \
  --data_dir data/synthetic \
  --epochs 5

echo "Training Multiscale FNO..."
bash scripts/train_multiscale.sh

echo "Optional: train diffusion model..."
bash scripts/train_diffusion.sh

echo "Now you can open visualization notebook or script."
echo "Example: notebooks/03_visualize_rollouts.ipynb"
