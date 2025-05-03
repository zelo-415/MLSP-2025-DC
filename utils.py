import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
 
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
     for input_tensor, hit_tensor, gt_tensor, mask_tensor in batch:
         pad_h = max_h - input_tensor.shape[1]
         pad_w = max_w - input_tensor.shape[2]
 
         input_p = F.pad(input_tensor, (0, pad_w, 0, pad_h), value=0)
         hit_p = F.pad(hit_tensor, (0, pad_w, 0, pad_h), value=1)
         gt_p = F.pad(gt_tensor, (0, pad_w, 0, pad_h), value=0)
         mask_p = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=1)
 
         padded_batch.append((input_p, hit_p, gt_p, mask_p))
 
     inputs, hits, gts, masks = zip(*padded_batch)
     return torch.stack(inputs), torch.stack(hits), torch.stack(gts), torch.stack(masks)