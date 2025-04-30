import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset1 import RadioMapDataset
from model import UNet
from utils import MSELoss, save_checkpoint, custom_collate_fn, plot_loss_curve
from pathlib import Path
from tqdm import tqdm
import os
import random
import numpy as np

# ==== Seed ====
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Dataset ====
full_dataset = RadioMapDataset(inputs_dir, outputs_dir, sparse_dir, positions_dir, los_dir=None, hit_dir=hit_dir)
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# ==== Model ====
model = UNet(in_channels=5, out_channels=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = MSELoss()

# ==== RMSE Accumulator ====
class RMSEAccumulator:
    def __init__(self):
        self.total_squared_error = 0.0
        self.total_pixels = 0

    def update(self, preds, targets, masks):
        se = ((preds - targets) * masks).pow(2)
        self.total_squared_error += se.sum().item()
        self.total_pixels += masks.sum().item()

    def compute(self):
        if self.total_pixels == 0:
            return float('inf')
        return (self.total_squared_error / self.total_pixels) ** 0.5

# ==== Training ====
best_val_rmse = float('inf')
train_rmses = []
val_rmses = []
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, epochs + 1):
    # ===== Train =====
    model.train()
    train_acc = RMSEAccumulator()
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

    for inputs, targets, masks in loop:
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        preds = model(inputs)
        loss = criterion(preds, targets, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc.update(preds, targets, masks)
        loop.set_postfix(train_loss=loss.item()**0.5)

    train_rmse = train_acc.compute()
    train_rmses.append(train_rmse)
    print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}")

    # ===== Val =====
    model.eval()
    val_acc = RMSEAccumulator()
    with torch.no_grad():
        for inputs, targets, masks in val_loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            preds = model(inputs)
            val_acc.update(preds, targets, masks)

    val_rmse = val_acc.compute()
    val_rmses.append(val_rmse)
    print(f"Epoch {epoch}: Val RMSE = {val_rmse:.4f}")

    # Save best model
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        save_checkpoint(model, "checkpoints/best_model.pth")
        print("Saved best model.")

# ==== Plot Loss Curves ====
plot_loss_curve(train_rmses, val_rmses, model_name=model.__class__.__name__)
