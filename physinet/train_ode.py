import argparse
import os

import torch
from torch.utils.data import DataLoader

from physinet.model.marine_fno import MarineFNO
from physinet.model.operator_ode import OperatorODE, integrate_trajectory
from physinet.data.real_data_loader import NPZWaveDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/marine")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="checkpoints/ode_fno.pt")
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = NPZWaveDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    base_operator = MarineFNO(modes=args.modes, width=args.width).to(device)
    ode_model = OperatorODE(base_operator)

    # contoh saja: integrasi satu trajectory
    batch = next(iter(loader)).to(device)  # [B, T,1,H,W]
    u0 = batch[:, 0, ...]                  # initial frame [B,1,H,W]

    t = torch.linspace(0.0, 1.0, steps=10, device=device)  # 10 time points
    traj = integrate_trajectory(ode_model, u0, t)          # [T,B,1,H,W]

    print("Integrated ODE trajectory shape:", traj.shape)

    torch.save(base_operator.state_dict(), args.save_path)
    print(f"Saved base MarineFNO operator to {args.save_path}")


if __name__ == "__main__":
    main()
