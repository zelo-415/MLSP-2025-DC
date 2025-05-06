import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dataset import RadioMapDataset
from model import UNet
import pandas as pd
from pathlib import Path

def visualize_error_map(sample_name, model_path, data_root):
    # === 初始化路径 ===
    inputs_dir = Path(data_root) / "inputs"
    outputs_dir = Path(data_root) / "outputs"
    sparse_dir = Path(data_root) / "sparse_samples_0.5"
    positions_dir = Path(data_root) / "Positions"
    hit_dir = Path(data_root) / "hitmap"

    # === 加载模型 ===
    model = UNet(in_channels=5, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # === 构造数据集并获取指定样本 ===
    dataset = RadioMapDataset(inputs_dir, outputs_dir, sparse_dir, positions_dir, hit_dir=hit_dir)
    idx = dataset.filenames.index(sample_name)
    input_tensor, gt_tensor, mask_tensor = dataset[idx]

    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0)
        pred_tensor = model(input_tensor).squeeze(0)

    # === 恢复单位并裁剪 padding ===
    H_gt, W_gt = gt_tensor.shape[1:]
    pred = pred_tensor[0, :H_gt, :W_gt].cpu().numpy() * 255
    gt = gt_tensor[0, :H_gt, :W_gt].cpu().numpy() * 255
    rgb = input_tensor[0, :3, :H_gt, :W_gt].cpu().numpy()

    # === 正确的墙体判断逻辑 ===
    wall_map = (rgb[0] != 0)

    # === 计算误差图 ===
    error_map = np.abs(pred - gt)

    # === 获取 TX 位置 ===
    sample_base = Path(sample_name).stem
    scene = "_".join(sample_base.split("_")[:-1])
    s_idx = int(sample_base.split("_")[-1][1:])
    pos_path = Path(positions_dir) / f"Positions_{scene}.csv"
    df = pd.read_csv(pos_path, index_col=0)
    tx_x, tx_y = int(df.loc[s_idx, "X"]), int(df.loc[s_idx, "Y"])

    # === 可视化：误差图 + 墙体边界 + TX位置 ===
    fig, ax = plt.subplots(figsize=(6, 6))

    # 显示误差图（增强对比）
    im = ax.imshow(error_map, cmap='hot', vmin=0, vmax=50)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|Pred - GT| (dB)")

    # 墙体轮廓叠加
    ax.contour(wall_map, colors='white', linewidths=0.5)

    # TX 位置叠加
    ax.scatter(tx_y, tx_x, c='cyan', s=50, marker='x', label='TX')
    ax.legend(loc="lower right")

    ax.set_title(f"误差图 + 墙体边界 + TX: {sample_name}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ==== 手动指定参数 ====
    sample_name = "B7_Ant1_f1_S15.png"
    model_path = "checkpoints/best_model.pth"
    data_root = "./"

    # ==== 调用可视化函数 ====
    visualize_error_map(sample_name, model_path, data_root)
