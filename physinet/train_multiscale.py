import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from physinet.models.ms_fno import MultiscaleFNO
from physinet.losses.physics_losses import wave_pde_residual
from physinet.data.real_data_loader import NPZWaveDataset


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    dt: float,
    c: float,
    lambda_phys: float,
):
    model.train()
    mse_loss = nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        # batch: [B, T, 1, H, W]
        batch = batch.to(device)
        # gunakan (t-1, t) -> pred t+1
        u_prev = batch[:, -3, ...]
        u_cur = batch[:, -2, ...]
        u_next = batch[:, -1, ...]

        optimizer.zero_grad()

        u_pred_next = model(u_cur)

        loss_data = mse_loss(u_pred_next, u_next)
        loss_phys = wave_pde_residual(u_prev, u_cur, u_pred_next, c=c, dt=dt)
        loss = loss_data + lambda_phys * loss_phys

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/marine")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lambda_phys", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="checkpoints/ms_fno.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device(args.device)

    dataset = NPZWaveDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MultiscaleFNO(modes=args.modes, width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            loader,
            optimizer,
            device,
            dt=args.dt,
            c=args.c,
            lambda_phys=args.lambda_phys,
        )
        print(f"[Epoch {epoch}] Loss = {loss:.6f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved multiscale FNO model to {args.save_path}")


if __name__ == "__main__":
    main()
