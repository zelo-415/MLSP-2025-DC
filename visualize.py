# visualize_prediction.py
import torch
import matplotlib.pyplot as plt
from dataset import RadioMapDataset
from model import UNet
from utils import custom_collate_fn, _convert_to_cartesian
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred_in_polar = True
data_root = Path("./competition/")
inputs_dir = data_root / "Inputs/Task_2_ICASSP"
outputs_dir = data_root / "Outputs/Task_2_ICASSP"
sparse_dir = data_root / "sparse_samples_0.5"
positions_dir = data_root / "Positions"

# === Load Dataset ===
dataset = RadioMapDataset(
    inputs_dir=inputs_dir,
    outputs_dir=outputs_dir,
    sparse_dir=sparse_dir,
    positions_dir=positions_dir
)

# Choose index manually
index = 721
input_tensor, gt_tensor, mask_tensor, (tx_x, tx_y) = dataset[index]
input_tensor = input_tensor.to(device)  # [1, C, H, W]
gt_tensor = gt_tensor.squeeze().cpu() * 255  # [H, W], dB
mask_tensor = mask_tensor.squeeze().cpu()

# === Load Model ===
model = UNet(in_channels=input_tensor.shape[1], out_channels=1).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()

# === Predict ===
with torch.no_grad():
    pred = model(input_tensor).squeeze().cpu() * 255  # [H, W], dB
    if pred_in_polar:
        pred = pred.unsqueeze(0)
        tx_x_i, tx_y_i = tx_x, tx_y
        print(f"tx_x_i: {tx_x_i}, tx_y_i: {tx_y_i}")
        H, W = pred.shape[1:]
        print(f"pred shape: {pred.shape}")
        pred = _convert_to_cartesian(pred.unsqueeze(0), (tx_x_i, tx_y_i), (H, W), device=None)
        pred = pred.squeeze().cpu()  # [H, W], dB
# === Compute error map ===
error_map = (gt_tensor - pred).abs()

# === Get RGB image ===
import torchvision.transforms as transforms
from PIL import Image
rgb = Image.open(inputs_dir/ dataset.filenames[index]).convert("RGB")
rgb = transforms.ToTensor()(rgb)[2].numpy()  # R channel only


# === Plot ===
fig, axes = plt.subplots(1, 6, figsize=(20, 4))
axes[0].imshow(rgb, cmap="gray")
axes[0].set_title("Input RGB (R channel)")
axes[1].imshow(input_tensor[0][2].cpu(), cmap="gray")
axes[1].set_title("input rgb (polar->cartesian)")
axes[2].imshow(gt_tensor, cmap="gray")
axes[2].set_title("Ground Truth (GT) [dB]")
axes[3].imshow(pred, cmap="gray")
axes[3].set_title("Prediction [dB]")
axes[4].imshow(mask_tensor, cmap="gray")
axes[4].set_title("Sampling Mask")
axes[5].imshow(error_map, cmap="hot")
axes[5].set_title("Absolute Error [dB]")

for ax in axes:
    ax.axis("off")

plt.savefig("plot_output.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()
