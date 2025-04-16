import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

def RMSELoss():
    """
    Calculates RMSE only on the non-sampled (masked-out) regions,
    and only within the valid (unpadded) image area.
    """
    def masked_rmse(pred, target, mask, valid_shape):
        h, w = valid_shape
        pred = pred[..., :h, :w]
        target = target[..., :h, :w]
        mask = mask[..., :h, :w]

        diff = ((pred - target) * 255) ** 2
        masked_diff = diff * (1 - mask)
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
    """
    Collate function to pad tensors to the same size and track original shape (h, w).
    Accepts batch of: (input_tensor, gt_tensor, mask_tensor, (h, w))
    """
    max_h = max(item[0].shape[1] for item in batch)
    max_w = max(item[0].shape[2] for item in batch)

    padded_batch = []
    valid_shapes = []

    for input_tensor, gt_tensor, mask_tensor, (h, w) in batch:
        pad_h = max_h - input_tensor.shape[1]
        pad_w = max_w - input_tensor.shape[2]

        input_p = F.pad(input_tensor, (0, pad_w, 0, pad_h), value=0)
        gt_p = F.pad(gt_tensor, (0, pad_w, 0, pad_h), value=0)
        mask_p = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=1.0)  # masked区域默认设为1

        padded_batch.append((input_p, gt_p, mask_p))
        valid_shapes.append((h, w))

    inputs, gts, masks = zip(*padded_batch)
    return torch.stack(inputs), torch.stack(gts), torch.stack(masks), valid_shapes
