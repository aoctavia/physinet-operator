import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from physinet.models.diffusion_model import DiffusionWaveModel
from physinet.data.real_data_loader import NPZWaveDataset


def cosine_beta_schedule(timesteps, s=0.008):
    import numpy as np

    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * (np.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.from_numpy(betas).float().clamp(0.0001, 0.9999)


def train_epoch(model, loader, optimizer, device, timesteps, betas, alphas_cumprod):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        # batch: [B, T,1,H,W], ambil frame terakhir sebagai target
        batch = batch.to(device)
        x0 = batch[:, -1, ...]  # [B,1,H,W]

        B = x0.size(0)
        t = torch.randint(0, timesteps, (B,), device=device).long()  # [B]
        noise = torch.randn_like(x0)

        # x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * noise
        alpha_t = alphas_cumprod[t].view(B, 1, 1, 1)
        sigma_t = torch.sqrt(1.0 - alpha_t)
        x_t = torch.sqrt(alpha_t) * x0 + sigma_t * noise

        optimizer.zero_grad()
        noise_pred = model(x_t, t.float() / timesteps)  # predict noise
        loss = mse(noise_pred, noise)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/marine")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="checkpoints/diffusion_wave.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = torch.device(args.device)

    dataset = NPZWaveDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = DiffusionWaveModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    betas = cosine_beta_schedule(args.timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            loader,
            optimizer,
            device,
            timesteps=args.timesteps,
            betas=betas,
            alphas_cumprod=alphas_cumprod,
        )
        print(f"[Epoch {epoch}] Diffusion loss = {loss:.6f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved diffusion model to {args.save_path}")


if __name__ == "__main__":
    main()
