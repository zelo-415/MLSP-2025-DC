# visualize_prediction.py
import torch
import matplotlib.pyplot as plt
from dataset1 import RadioMapDataset
from model import UNet
from utils import custom_collate_fn
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Dataset ===
dataset = RadioMapDataset(
    inputs_dir="inputs",
    outputs_dir="outputs",
    sparse_dir="sparse_samples_0.5",
    positions_dir="Positions"
)

# Choose index manually
index = 50
input_tensor, gt_tensor, mask_tensor = dataset[index]
input_tensor = input_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
gt_tensor = gt_tensor.squeeze().cpu() * 255  # [H, W], dB
mask_tensor = mask_tensor.squeeze().cpu()

# === Load Model ===
model = UNet(in_channels=input_tensor.shape[1], out_channels=1).to(device)
model.load_state_dict(torch.load("checkpoints/bestbestmodel.pth", map_location=device))
model.eval()

# === Predict ===
with torch.no_grad():
    pred = model(input_tensor).squeeze().cpu() * 255  # [H, W], dB

# === Compute error map ===
error_map = (gt_tensor - pred).abs()

# === Get RGB image ===
import torchvision.transforms as transforms
from PIL import Image
rgb = Image.open(Path("inputs") / dataset.filenames[index]).convert("RGB")
rgb = transforms.ToTensor()(rgb)[0].numpy()  # R channel only

# === Plot ===
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(rgb, cmap="gray")
axes[0].set_title("Input RGB (R channel)")
axes[1].imshow(gt_tensor, cmap="gray")
axes[1].set_title("Ground Truth (GT) [dB]")
axes[2].imshow(pred, cmap="gray")
axes[2].set_title("Prediction [dB]")
axes[3].imshow(mask_tensor, cmap="gray")
axes[3].set_title("Sampling Mask")
axes[4].imshow(error_map, cmap="hot")
axes[4].set_title("Absolute Error [dB]")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
