import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import os
import torch.nn.functional as F
import pandas as pd
import random  # Add this import

from utils import convert_to_polar, convert_to_cartesian, find_FSPL
 
class RadioMapDataset(Dataset):
    def __init__(self, inputs_dir, outputs_dir, sparse_dir, positions_dir, los_dir = None, hit_dir = None):
        self.inputs_dir = Path(inputs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.sparse_dir = Path(sparse_dir)
        self.positions_dir = Path(positions_dir)
        self.los_dir = Path(los_dir) if los_dir else None
        self.hit_dir = Path(hit_dir) if hit_dir else None

        self.filenames = sorted([f.name for f in self.inputs_dir.glob("*.png")])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load RGB (3-channel physical input)
        rgb = Image.open(self.inputs_dir / fname).convert("RGB")
        rgb_tensor = self.to_tensor(rgb)  # [3, H, W]
        C, H, W = rgb_tensor.shape
        center = (W // 2, H // 2)
        rgb_tensor[0] = 255 * rgb_tensor[0] / 20
        rgb_tensor[1] = 255 * rgb_tensor[1] / 40
        rgb_tensor[2] = torch.log10(1 + 255 * rgb_tensor[2]) / 2.5
        
        # Load GT PL map (grayscale)
        gt = Image.open(self.outputs_dir / fname).convert("L")
        gt_tensor = self.to_tensor(gt)

        # Load sparse samples (x, y, pl)
        sparse_points = np.load(self.sparse_dir / (Path(fname).stem + ".npy"))
        h, w = gt_tensor.shape[1:]
        sparse_map = torch.zeros((1, h, w))
        mask_map = torch.zeros((1, h, w))
        for x, y, pl in sparse_points:
            sparse_map[0, int(y), int(x)] = pl / 100.0  # Normalization
            mask_map[0, int(y), int(x)] = 1.0

        #print(f"sparse_map: {sparse_map.max()}, {sparse_map.min()}")
        
        if self.hit_dir:
            hit_fname = Path(fname).stem + "_hit.npy"
            hit_path = self.hit_dir / hit_fname
            hit_map = np.load(hit_path)  # [H, W]
            
            if np.max(hit_map) > 0:
                hit_map = hit_map / np.max(hit_map)
            else:
                hit_map = np.zeros_like(hit_map)

            hit_tensor = torch.from_numpy(hit_map).unsqueeze(0).float()  # [1, H, W]
        else:
            hit_tensor = torch.zeros((1, h, w)).float()

        
        if random.random() < .4 and H*W > 60e3:
            tx_y, tx_x = self._load_tx_xy(fname)
            crop_size_x = W // 2  # Define the crop size (e.g., 128x128)
            crop_size_y = H // 2  # Define the crop size (e.g., 128x128)
            x_min = max(0, int(tx_x - crop_size_x // 2))
            y_min = max(0, int(tx_y - crop_size_y // 2))
            x_max = min(w, x_min + crop_size_x)
            y_max = min(h, y_min + crop_size_y)

            # Crop all tensors
            rgb_tensor = rgb_tensor[:, y_min:y_max, x_min:x_max]
            gt_tensor = gt_tensor[:, y_min:y_max, x_min:x_max]
            sparse_map = sparse_map[:, y_min:y_max, x_min:x_max]
            mask_map = mask_map[:, y_min:y_max, x_min:x_max]
            hit_tensor = hit_tensor[:, y_min:y_max, x_min:x_max]

        # Concatenate tensors
        input_tensor = torch.cat([rgb_tensor, sparse_map], dim=0)
        input_tensor, hit_tensor, gt_tensor, mask_map = self.pad_all(input_tensor, hit_tensor, gt_tensor, mask_map)
        input_tensor = torch.cat([input_tensor, hit_tensor], dim=0)

        return input_tensor, gt_tensor, mask_map

    def pad_all(self, input_tensor, hit_tensor, gt_tensor, mask_tensor):
        _, h, w = input_tensor.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        hit_tensor = F.pad(hit_tensor, (0, pad_w, 0, pad_h), mode='constant', value=1)
        gt_tensor = F.pad(gt_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='constant', value=1)

        return input_tensor, hit_tensor, gt_tensor, mask_tensor

    def _load_tx_xy(self, fname):
        name = Path(fname).stem
        scene = "_".join(name.split("_")[:-1])
        s_idx = int(name.split("_")[-1][1:])
        pos_path = Path(self.positions_dir) / f"Positions_{scene}.csv"
        df = pd.read_csv(pos_path)
        tx_x, tx_y = int(df.loc[s_idx, "X"]), int(df.loc[s_idx, "Y"])
        return tx_x, tx_y