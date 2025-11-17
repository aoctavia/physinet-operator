#!/bin/bash

echo "Generating dataset..."
python -m physigym.env --output_dir data/synthetic --num_trajectories 20 --pde_type wave2d

echo "Training model..."
python -m physinet.train --data_dir data/synthetic --epochs 5

echo "Open notebook: notebooks/03_visualize_rollouts.ipynb"
