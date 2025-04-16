import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import pandas as pd

class RadioMapDataset(Dataset):
    def __init__(self, inputs_dir, outputs_dir, sparse_dir, positions_dir):
        self.inputs_dir = Path(inputs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.sparse_dir = Path(sparse_dir)
        self.positions_dir = Path(positions_dir)
        self.filenames = sorted([f.name for f in self.inputs_dir.glob("*.png")])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # === Load RGB input ===
        rgb = Image.open(self.inputs_dir / fname).convert("RGB")
        rgb_tensor = self.to_tensor(rgb)  # [3, H, W]
        rgb_tensor[2] = 1.0 - rgb_tensor[2]  # Flip blue channel

        # === Load Ground Truth ===
        gt = Image.open(self.outputs_dir / fname).convert("L")
        gt_tensor = self.to_tensor(gt)  # [1, H, W] (still normalized in 0-1 range)

        # === Load sparse samples ===
        sparse_points = np.load(self.sparse_dir / (Path(fname).stem + ".npy"))
        h, w = gt_tensor.shape[1:]
        sparse_map = torch.zeros((1, h, w))
        mask_map = torch.ones((1, h, w))  # default: masked (1)

        for x, y, pl in sparse_points:
            sparse_map[0, int(y), int(x)] = pl / 100.0  # normalize
            mask_map[0, int(y), int(x)] = 0.0  # unmasked = supervised

        # === Load Tx position and encode as Gaussian heatmap ===
        pos_file = self._find_position_file(fname)
        tx_x, tx_y = self._load_tx_xy(pos_file)
        heatmap = self._generate_gaussian_heatmap(tx_x, tx_y, h, w, sigma=25.0).unsqueeze(0)

        # === Stack input channels ===
        input_tensor = torch.cat([rgb_tensor, sparse_map, heatmap], dim=0)  # [5, H, W]

        # === Pad all tensors ===
        input_tensor, gt_tensor, mask_map = self.pad_all(input_tensor, gt_tensor, mask_map)

        # === Return original size for loss crop ===
        return input_tensor, gt_tensor, mask_map, (h, w)

    def pad_all(self, input_tensor, gt_tensor, mask_tensor):
        _, h, w = input_tensor.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        gt_tensor = F.pad(gt_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='constant', value=1.0)  # default: masked
        return input_tensor, gt_tensor, mask_tensor

    def _find_position_file(self, fname):
        base = Path(fname).stem
        building = base.split("_")[0]
        pos_file = list(self.positions_dir.glob(f"Positions_{building}_*.csv"))
        if not pos_file:
            raise FileNotFoundError(f"No position file found for {fname}")
        return pos_file[0]

    def _load_tx_xy(self, filepath):
        df = pd.read_csv(filepath)
        return float(df.iloc[0]['X']), float(df.iloc[0]['Y'])

    def _generate_gaussian_heatmap(self, tx_x, tx_y, h, w, sigma=15.0):
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        d2 = (xx - tx_x)**2 + (yy - tx_y)**2
        return torch.exp(-d2 / (2 * sigma**2))
