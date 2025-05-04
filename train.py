import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import RadioMapDataset
from model import UNet
from utils import MSELoss, compute_rmse, save_checkpoint, custom_collate_fn, plot_loss_curve
from pathlib import Path
from tqdm import tqdm
import os
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

set_seed(42)

# ==== Config ====
data_root = Path("./")
inputs_dir = data_root / "inputs"
outputs_dir = data_root / "outputs"
sparse_dir = data_root / "sparse_samples_0.5"
positions_dir = data_root / "Positions"
los_dir = data_root / "losmap"
hit_dir = data_root / "hitmap"

batch_size = 4
epochs = 50
lr = 1e-4
val_ratio = 0.2
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# ==== Load dataset ====
full_dataset = RadioMapDataset(inputs_dir, outputs_dir, sparse_dir, positions_dir, los_dir=None, hit_dir=hit_dir)
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

example_input, example_target, example_mask = train_set[0]
C, H, W = example_input.shape

# ==== Initialize model ====
model = UNet(in_channels=C, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = MSELoss()

# ==== Training loop ====
best_val_rmse = float('inf')
train_losses = []
val_losses = []

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, epochs + 1):
    model.train()
    train_squared_error_sum = 0.0
    train_pixel_count = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    for inputs, targets, masks in loop:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        preds = model(inputs)
        mse_loss = criterion(preds, targets, masks)

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        batch_error = ((preds - targets) * 255) ** 2
        batch_error = batch_error * (1 - masks)
        train_squared_error_sum += batch_error.sum().item()
        train_pixel_count += (1 - masks).sum().item()

        loop.set_postfix(train_rmse=(mse_loss.item())**0.5)

    
    avg_train_rmse = (train_squared_error_sum / train_pixel_count) ** 0.5
    train_losses.append(avg_train_rmse)
    print(f"Epoch {epoch}: Train RMSE = {avg_train_rmse:.4f}")

    model.eval()
    val_squared_error_sum = 0.0
    val_pixel_count = 0

    with torch.no_grad():
        for inputs, targets, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            preds = model(inputs)
            rmse_batch = compute_rmse(preds, targets, masks)

            mse_batch = rmse_batch.item()
            num_valid_pixels = (1 - masks).sum().item()
            val_squared_error_sum += mse_batch * num_valid_pixels
            val_pixel_count += num_valid_pixels

    avg_val_rmse = (val_squared_error_sum / val_pixel_count) ** 0.5
    val_losses.append(avg_val_rmse)
    print(f"Epoch {epoch}: Val RMSE = {avg_val_rmse:.4f}")

    if avg_val_rmse < best_val_rmse:
        best_val_rmse = avg_val_rmse
        save_checkpoint(model, "checkpoints/best_model.pth")
        print("Saved best model.")

# Visualize loss curves
plot_loss_curve(train_losses, val_losses, "checkpoints/loss_curve.png")