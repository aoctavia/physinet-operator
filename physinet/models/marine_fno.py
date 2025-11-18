import torch
import torch.nn as nn

from .sea_load_head import SeaLoadHead


class MarineFNO(nn.Module):
    def __init__(self, modes: int = 12, width: int = 64, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.modes = modes
        self.width = width

        # CONTOH: kamu pasti sudah punya definisi FNO sendiri
        # Sesuaikan bagian ini dengan kode kamu sekarang
        self.fc0 = nn.Linear(in_channels, width)
        # ... layer Fourier / conv berikutnya ...

        self.fc_out = nn.Linear(width, out_channels)

        # NEW: sea load head
        self.sea_load_head = SeaLoadHead(in_channels=width, hidden=64, out_dim=3)

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        return:
          u_pred: [B, 1, H, W]
          sea_loads: [B, 3] (Fx,Fy,Mz)
        """
        B, C, H, W = x.shape
        # contoh pipeline, sesuaikan dengan punyamu:
        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        x = self.fc0(x)            # [B,H,W,width]
        # ... FNO layers ...
        latent = x.permute(0, 3, 1, 2)  # [B,width,H,W]

        # output field
        out = latent.permute(0, 2, 3, 1)  # [B,H,W,width]
        out = self.fc_out(out)            # [B,H,W,1]
        u_pred = out.permute(0, 3, 1, 2)  # [B,1,H,W]

        # NEW: sea loads
        sea_loads = self.sea_load_head(latent)  # [B,3]

        return u_pred, sea_loads
