import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dataset import RadioMapDataset
from model import UNet
import pandas as pd
from pathlib import Path

def visualize_error_map(sample_name, model_path, data_root):
    
    inputs_dir = Path(data_root) / "inputs"
    outputs_dir = Path(data_root) / "outputs"
    sparse_dir = Path(data_root) / "sparse_samples_0.5"
    positions_dir = Path(data_root) / "Positions"
    hit_dir = Path(data_root) / "hitmap"


    model = UNet(in_channels=5, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    dataset = RadioMapDataset(inputs_dir, outputs_dir, sparse_dir, positions_dir, hit_dir=hit_dir)
    idx = dataset.filenames.index(sample_name)
    input_tensor, gt_tensor, mask_tensor = dataset[idx]

    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0)
        pred_tensor = model(input_tensor).squeeze(0)

    H_gt, W_gt = gt_tensor.shape[1:]
    pred = pred_tensor[0, :H_gt, :W_gt].cpu().numpy() * 255
    gt = gt_tensor[0, :H_gt, :W_gt].cpu().numpy() * 255
    rgb = input_tensor[0, :3, :H_gt, :W_gt].cpu().numpy()
    wall_map = (rgb[0] != 0)
    sample_base = Path(sample_name).stem
    scene = "_".join(sample_base.split("_")[:-1])
    s_idx = int(sample_base.split("_")[-1][1:])
    pos_path = Path(positions_dir) / f"Positions_{scene}.csv"
    df = pd.read_csv(pos_path, index_col=0)
    tx_x, tx_y = int(df.loc[s_idx, "X"]), int(df.loc[s_idx, "Y"])

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(error_map, cmap='hot', vmin=0, vmax=50)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|Pred - GT| (dB)")

    ax.contour(wall_map, colors='white', linewidths=0.5)

    ax.scatter(tx_y, tx_x, c='cyan', s=50, marker='x', label='TX')
    ax.legend(loc="lower right")

    ax.set_title(f"wall and error: {sample_name}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
   
    sample_name = "B7_Ant1_f1_S15.png"
    model_path = "checkpoints/best_model.pth"
    data_root = "./"
    visualize_error_map(sample_name, model_path, data_root)
