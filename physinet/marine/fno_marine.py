import torch
import torch.nn as nn
import torch.fft

class MarineFNO(nn.Module):
    """
    Minimal Fourier Neural Operator for marine wave prediction.
    Input: initial condition η(x,y,t0)
    Output: future wave field η(x,y,t)
    """

    def __init__(self, modes=12, width=32):
        super().__init__()
        self.modes = modes
        self.width = width

        self.fc0 = nn.Linear(1, width)

        self.conv = SpectralConv2d(width, width, modes, modes)
        self.w = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc0(x)  # [B,H,W,1] → [B,H,W,width]
        x = x.permute(0, 3, 1, 2)  # → [B,width,H,W]

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = torch.relu(x)

        x = x.permute(0, 2, 3, 1)  # back to [B,H,W,width]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SpectralConv2d(nn.Module):
    """Stable spectral convolution for Fourier Neural Operator."""

    def __init__(self, in_ch, out_ch, modes_x, modes_y):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Complex weights
        self.scale = 1 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_ch, out_ch, modes_x, modes_y, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Fourier transform
        x_ft = torch.fft.rfft2(x)   # shape: [B,C,H,W//2+1]

        out_ft = torch.zeros(
            B, self.out_ch, H, W//2 + 1, dtype=torch.cfloat, device=x.device
        )

        # Only use the lower modes
        mx = min(self.modes_x, H)
        my = min(self.modes_y, W//2 + 1)

        out_ft[:, :, :mx, :my] = torch.einsum(
            "bchw,ciow->bioh",
            x_ft[:, :, :mx, :my],
            self.weights[:, :, :mx, :my]
        ).permute(0, 1, 2, 3)

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x
