#!/bin/bash

echo "==========================================="
echo " Training Marine Fourier Neural Operator "
echo "==========================================="

# Activate env (opsional, jika kamu pakai venv)
# source venv/bin/activate

echo "[1] Generating synthetic marine dataset..."
python3 - <<EOF
from physinet.marine.gym import MarineWaveGym
import numpy as np

gym = MarineWaveGym(nx=64, ny=64, nt=20)
data = gym.rollout_dataset(50)

np.savez("data/marine_wave_dataset.npz", data=data)
print("Saved to data/marine_wave_dataset.npz")
EOF

echo "[2] Training MarineFNO model..."
python3 - <<EOF
import torch
import torch.nn as nn
import numpy as np

from physinet.marine.fno_marine import MarineFNO

dataset = np.load("data/marine_wave_dataset.npz", allow_pickle=True)["data"]

inputs, targets = [], []
for sample in dataset:
    eta = sample["eta"]
    x0 = eta[:,:,0]
    xt = eta[:,:,10]
    inputs.append(x0[...,None])
    targets.append(xt[...,None])

inputs = torch.tensor(np.stack(inputs), dtype=torch.float32)
targets = torch.tensor(np.stack(targets), dtype=torch.float32)

model = MarineFNO(modes=12, width=32)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(5):
    opt.zero_grad()
    pred = model(inputs)
    loss = loss_fn(pred, targets)
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1}, Loss = {loss.item():.6f}")

torch.save(model.state_dict(), "models/marine_fno.pt")
print("Saved model to models/marine_fno.pt")
EOF

echo "Done!"
