import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

from data_loader import create_dataloader
from unet_model import UNet, UNetWithAttention

# ==== Config ====
input_path = "ICASSP2025_Dataset/Inputs/Task_2_ICASSP/"
output_path = "ICASSP2025_Dataset/Outputs/Task_2_ICASSP/"

all_file_names = [f"B{b}_Ant1_f1_S{s}" for b in range(1, 6) for s in range(10)]
train_files, val_files = train_test_split(all_file_names, test_size=0.2, random_state=42)

train_loader = create_dataloader(train_files, input_path, output_path, batch_size=2)
val_loader = create_dataloader(val_files, input_path, output_path, batch_size=1, shuffle=False)

# dataloader = create_dataloader(file_names, input_path, output_path, batch_size=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet().to(device)
model = UNetWithAttention().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def masked_rmse(pred, target, mask):
    mse = ((pred - target) ** 2 * (1 - mask)).sum() / (1 - mask).sum()
    return torch.sqrt(mse)

def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            loss = masked_rmse(pred, y, mask)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# ==== Train over multiple epochs ====
epochs = 20
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        x, y, mask = [b.to(device) for b in batch]
        pred = model(x)
        loss = masked_rmse(pred, y, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    val_rmse = evaluate(model, val_loader, device)
    val_losses.append(val_rmse)
    print(f"Epoch {epoch+1}: Train RMSE = {avg_loss:.4f} → Val RMSE = {val_rmse:.4f}")


# ==== Plot loss curve ====
plt.figure()
plt.plot(range(1, epochs+1), train_losses, label="Train")
plt.plot(range(1, epochs+1), val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Masked RMSE")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save with timestamp
os.makedirs("plots", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"plots/loss_curve_{timestamp}.png")
plt.show()
