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
     for input_tensor, gt_tensor, mask_tensor, txs in batch:
         pad_h = max_h - input_tensor.shape[1]
         pad_w = max_w - input_tensor.shape[2]
 
         input_p = F.pad(input_tensor, (0, pad_w, 0, pad_h), value=0)
         gt_p = F.pad(gt_tensor, (0, pad_w, 0, pad_h), value=0)
         mask_p = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=0)
         txs = torch.tensor(txs, dtype=torch.float32)
         padded_batch.append((input_p, gt_p, mask_p, txs))
 
     inputs, gts, masks, txs = zip(*padded_batch)
     return torch.stack(inputs), torch.stack(gts), torch.stack(masks), torch.stack(txs)



def _convert_to_polar(tensor: torch.Tensor, center: tuple[int, int], num_radial=None, num_angles=None):
    """
    Convert a batch of images from Cartesian to Polar coordinates.

    Args:
        tensor: (B, C, H, W)
        center: (cy, cx)
        num_radial: optional number of radial steps
        num_angles: optional number of angular steps

    Returns:
        (B, C, num_radial, num_angles)
    """
    assert tensor.ndim == 4, "Input tensor must have shape (B, C, H, W)"
    B, C, H, W = tensor.shape
    cy, cx = center

    if num_radial is None:
        num_radial = int(math.hypot(max(cy, H-cy), max(cx, W-cx)))
    if num_angles is None:
        num_angles = 360

    device = tensor.device

    # Generate radial and angular coordinates
    max_x = max(cx, W-cx)
    max_y = max(cy, H-cy)
    max_radius = math.hypot(max_x, max_y)

    r = torch.linspace(0, 1, steps=num_radial, device=device) * max_radius
    theta = torch.linspace(0, 2 * math.pi, steps=num_angles, device=device)

    rr, tt = torch.meshgrid(r, theta, indexing='ij')  # (num_radial, num_angles)

    # Polar to Cartesian
    x = rr * torch.cos(tt)
    y = rr * torch.sin(tt)

    # Scale and shift to match center
    x = x + cx
    y = y + cy

    # Normalize to [-1, 1] for grid_sample
    grid_x = (2 * x / (W - 1)) - 1
    grid_y = (2 * y / (H - 1)) - 1

    grid = torch.stack((grid_x, grid_y), dim=-1)  # (num_radial, num_angles, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)   # (B, num_radial, num_angles, 2)

    # Sample
    polar = F.grid_sample(tensor, grid, mode='bicubic', align_corners=False)

    return polar  # (B, C, num_radial, num_angles)

def _convert_to_cartesian(polar_tensor: torch.Tensor, center: tuple[int, int], output_size: tuple[int, int], device=None):
    assert polar_tensor.ndim == 4, "Input polar_tensor must have shape (B, C, num_radial, num_angles)"
    B, C, num_radial, num_angles = polar_tensor.shape
    H, W = output_size
    cy, cx = center

    if device is None:
        device = polar_tensor.device

    # Create Cartesian grid
    y = torch.linspace(0, H-1, H, device=device)
    x = torch.linspace(0, W-1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)

    xx = xx - cx
    yy = yy - cy

    # Cartesian to Polar
    r = torch.sqrt(xx**2 + yy**2)
    theta = torch.atan2(yy, xx)
    theta = theta % (2 * math.pi)

    # --- FIX: match max_radius calculation exactly like in _convert_to_polar ---
    max_radius = math.hypot(max(cx, W-cx), max(cy, H-cy))

    # Normalize
    r_norm = r / max_radius
    theta_norm = theta / (2 * math.pi)

    # Scale normalized coordinates to polar tensor size
    r_idx = r_norm * (num_radial - 1)
    theta_idx = theta_norm * (num_angles - 1)

    # Normalize to [-1, 1] for grid_sample
    r_idx = (2 * r_idx / (num_radial - 1)) - 1
    theta_idx = (2 * theta_idx / (num_angles - 1)) - 1

    # --- FIX: swap axis order for grid_sample ---
    grid = torch.stack((theta_idx, r_idx), dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)     # (B, H, W, 2)

    # Sample
    cartesian = F.grid_sample(polar_tensor, grid, mode='bicubic', align_corners=True, padding_mode='border')

    return cartesian  # (B, C, H, W)

def find_FSPL(fname, X):
     freqs_GHz = [0.868, 1.8, 3.5]
     match = re.search(r'_f(\d+)', fname)
     fnum = int(match.group(1))
     freq_GHz = freqs_GHz[fnum-1]
     lam = 0.3 / freq_GHz
     fspl =  ( (4 * math.pi * X) + 1e-6 / lam )
     fspl = 20 * np.log10 ( fspl )
     #fspl = torch.tensor(fspl, dtype = torch.float32)
     return fspl

