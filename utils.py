import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import math
 
def RMSELoss():
     def masked_rmse(pred, target, mask):
         diff = ((pred - target)*255) ** 2
         masked_diff = diff * (1 - mask)  # Only penalize on non-sampled locations
         mse = masked_diff.sum() / (1 - mask).sum().clamp(min=1.0)
         return torch.sqrt(mse)
     return masked_rmse
 
def save_checkpoint(model, path):
     os.makedirs(os.path.dirname(path), exist_ok=True)
     torch.save(model.state_dict(), path)
 
def plot_loss_curve(train_losses, val_losses, out_path):
     os.makedirs(os.path.dirname(out_path), exist_ok=True)
     plt.figure()
     plt.plot(train_losses, label='Train Loss')
     plt.plot(val_losses, label='Val Loss')
     plt.xlabel('Epoch')
     plt.ylabel('RMSE Loss')
     plt.title('Training vs Validation Loss')
     plt.legend()
     plt.grid(True)
     plt.savefig(out_path)
     plt.close()
 
def custom_collate_fn(batch):
     # batch: List of (input, target, mask)
     max_h = max(item[0].shape[1] for item in batch)
     max_w = max(item[0].shape[2] for item in batch)
 
     padded_batch = []
     for input_tensor, gt_tensor, mask_tensor in batch:
         pad_h = max_h - input_tensor.shape[1]
         pad_w = max_w - input_tensor.shape[2]
 
         input_p = F.pad(input_tensor, (0, pad_w, 0, pad_h), value=0)
         gt_p = F.pad(gt_tensor, (0, pad_w, 0, pad_h), value=0)
         mask_p = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=0)
 
         padded_batch.append((input_p, gt_p, mask_p))
 
     inputs, gts, masks = zip(*padded_batch)
     return torch.stack(inputs), torch.stack(gts), torch.stack(masks)

def _convert_to_polar(tensor: torch.Tensor, center: tuple[int, int], num_radial=None, num_angles=None):
     """
     Converts a (C, H, W) tensor to polar coordinates per channel.

     Args:
          tensor (torch.Tensor): Input tensor of shape (C, H, W)
          center (tuple[int, int]): The (x, y) center of the polar transform
          num_radial (int, optional): Number of radial steps (defaults to half of min(H, W))
          num_angles (int, optional): Number of angular steps (defaults to 360)

     Returns:
          torch.Tensor: Polar-transformed tensor of shape (C, num_radial, num_angles)
     """
     C, H, W = tensor.shape
     cx, cy = center

     if num_radial is None:
          num_radial = min(H, W)
     if num_angles is None:
          num_angles = 360

     # Create polar coordinate grid (r, θ)
     max_x = max(cx, W - cx)
     max_y = max(cy, H - cy)
     max_r = int( (max_x**2 + max_y**2) ** 0.5 )

     r = torch.linspace(0, 1, steps=num_radial) * max_r
     theta = torch.linspace(0, 2 * torch.pi, steps=num_angles)
     grid_r, grid_theta = torch.meshgrid(r, theta, indexing='ij')

     # Convert polar to cartesian (normalized to [-1, 1] for grid_sample)
     x = grid_r * torch.cos(grid_theta) + cx
     y = grid_r * torch.sin(grid_theta) + cy

     x_norm = 2 * x / (W - 1) - 1
     y_norm = 2 * y / (H - 1) - 1

     grid = torch.stack((x_norm, y_norm), dim=-1)  # (num_radial, num_angles, 2)

     # Reshape tensor to (1, C, H, W) and grid to (1, num_radial, num_angles, 2)
     tensor = tensor.unsqueeze(0)
     grid = grid.unsqueeze(0)

     # grid_sample expects (N, C, H, W) and grid in (N, H_out, W_out, 2)
     polar_tensor = F.grid_sample(tensor, grid, mode='bilinear', align_corners=True)

     return polar_tensor.squeeze(0).permute(0, 2, 1)

def find_FSPL(fname, X):
     X = X.numpy()
     freqs_GHz = [0.868, 1.8, 3.5]
     match = re.search(r'_f(\d+)', fname)
     fnum = int(match.group(1))
     freq_GHz = freqs_GHz[fnum-1]
     lam = 0.3 / freq_GHz
     fspl =  ( (4 * math.pi * X[:, :, 2]) + 1e-6 / lam )
     fspl = 20 * np.log10 ( fspl )
     fspl = torch.tensor(fspl, dtype = torch.float32)
     return fspl