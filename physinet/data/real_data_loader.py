import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class NPZWaveDataset(Dataset):
    """
    Simple dataset for .npz files with key 'u' -> [T, H, W].
    You can adapt to metocean/lab data later.
    """

    def __init__(self, root_dir: str, time_stride: int = 1):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        self.time_stride = time_stride

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path)
        u = data["u"]  # [T, H, W]
        u = torch.from_numpy(u).float()  # [T,H,W]
        # add channel dimension
        u = u.unsqueeze(1)  # [T,1,H,W]
        return u
