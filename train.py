import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import RadioMapDataset
from model import UNet
# from unet_seblock import UNetWithSE
# from unet_freq import UNet
from utils import RMSELoss, save_checkpoint, custom_collate_fn, plot_loss_curve 
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load dataset ====
full_dataset = RadioMapDataset(inputs_dir, outputs_dir, sparse_dir, positions_dir, los_dir = None, hit_dir = hit_dir)
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# ==== Initialize model ====
model = UNet(in_channels=5, out_channels=1).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
criterion = RMSELoss()

# ==== Training loop ====
best_val_loss = float('inf')
train_losses = []  
val_losses = []    

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    for inputs, targets, masks in loop:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        # freq = freq.to(device)

        preds = model(inputs)
        loss = criterion(preds, targets, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        train_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  #
    print(f"Epoch {epoch}: Train RMSE = {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets, masks in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            # freq = freq.to(device)

            preds = model(inputs)
            loss = criterion(preds, targets, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  
    print(f"Epoch {epoch}: Val RMSE = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_checkpoint(model, "checkpoints/best_model.pth")
        print("Saved best model.")

# Visualize loss curves
plot_loss_curve(train_losses, val_losses, model_name=model.__class__.__name__)

