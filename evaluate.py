import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from model import UNet
from hit_batch1 import generate_wall_mask, generate_hit_map
from utils import find_FSPL
import matplotlib.pyplot as plt  # Add this import


def load_sparse_png(png_path):
    img = Image.open(png_path).convert("L")
    sparse = np.array(img).astype(np.float32)
    sparse = sparse / 100.0
    return torch.from_numpy(sparse).unsqueeze(0)  # [1, H, W]

def prepare_input(rgb_path, sparse_path, positions_dir):
    name = rgb_path.stem
    scene = "_".join(name.split("_")[:-1])
    s_idx = int(name.split("_")[-1][1:])
    pos_path = Path(positions_dir) / f"Positions_{scene}.csv"
    df = pd.read_csv(pos_path)
    tx_x, tx_y = int(df.loc[s_idx, "X"]), int(df.loc[s_idx, "Y"])

    # RGB 预处理
    rgb = Image.open(rgb_path).convert("RGB")
    rgb_tensor = transforms.ToTensor()(rgb)
    rgb_tensor[0] = 255 * rgb_tensor[0] / 20
    rgb_tensor[1] = 255 * rgb_tensor[1] / 40
    rgb_tensor[2] = torch.log10(1 + 255 * rgb_tensor[2]) / 2.5


    sparse_tensor = load_sparse_png(sparse_path)  # 注意这里不再除以100
    # hitmap 生成
    wall_mask, d = generate_wall_mask(rgb_path)
    hit_map = generate_hit_map(wall_mask, tx_x, tx_y)
    fspl = find_FSPL(name, torch.tensor(d).float()).numpy()
    hit_map = fspl+hit_map
    hit_map = hit_map / np.max(hit_map)
    hit_tensor = torch.from_numpy(hit_map).unsqueeze(0).float()

    # 合并 & padding
    input_tensor = torch.cat([rgb_tensor, sparse_tensor], dim=0)
    _, h, w = input_tensor.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    hit_tensor = F.pad(hit_tensor, (0, pad_w, 0, pad_h), mode='constant', value=1)
    input_tensor = torch.cat([input_tensor, hit_tensor], dim=0)

    return input_tensor.unsqueeze(0), h, w, name  # 返回完整 name

def flatten_output(output, h, w, name):
    output = output.squeeze().cpu().numpy()
    output = output[:h, :w] * 255.0
    output = np.clip(output, 13, 160)
    ids = [f"{name}_{i}" for i in range(output.size)]
    return ids, output.flatten().tolist()

def save_visualizations(inputs, outputs, h, w, sample_names, output_dir, num_samples=5):
    """
    Save visualizations of input and output samples as PNG files.
    :param inputs: Tensor of input samples (B, C, H, W)
    :param outputs: Tensor of output samples (B, 1, H, W)
    :param h: Original height of the samples
    :param w: Original width of the samples
    :param sample_names: List of sample names
    :param output_dir: Directory to save the visualizations
    :param num_samples: Number of samples to visualize
    """
    inputs = inputs.cpu().numpy()
    outputs = outputs.cpu().numpy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(num_samples, inputs.shape[0])):
        input_sample = inputs[i]
        output_sample = outputs[i].squeeze()

        # Plot all input channels
        num_channels = input_sample.shape[0]
        fig, axes = plt.subplots(1, num_channels + 1, figsize=(15, 5))
        fig.suptitle(f"Sample: {sample_names[i]}")

        for c in range(num_channels):
            axes[c].imshow(input_sample[c], cmap="gray")
            axes[c].set_title(f"Input Channel {c+1}")
            axes[c].axis("off")

        # Plot the output
        axes[-1].imshow(output_sample[:h, :w], cmap="viridis")
        axes[-1].set_title("Output")
        axes[-1].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"{sample_names[i]}.png")
        plt.close(fig)

@torch.no_grad()
def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=5, out_channels=1).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()

    input_dir = Path("./test_input")
    sparse_dir = Path("./rate0.5/sampledGT")
    pos_dir = Path("./test_positions")
    visualization_dir = "./visualizations"  # Directory to save visualizations

    all_ids, all_pls = [], []
    input_samples, output_samples, sample_names = [], [], []

    for idx, rgb_path in enumerate(tqdm(sorted(input_dir.glob("*.png")))):
        sparse_path = sparse_dir / (rgb_path.stem + ".png")
        input_tensor, h, w, name = prepare_input(rgb_path, sparse_path, pos_dir)
        input_tensor = input_tensor.to(device)

        pred = model(input_tensor)
        ids, pls = flatten_output(pred, h, w, name)
        all_ids.extend(ids)
        all_pls.extend(pls)

        # Collect samples for visualization
        if idx > 1 and idx < 7:  # Limit to 5 samples
            input_samples.append(input_tensor[0])  # First sample in the batch
            output_samples.append(pred[0])  # First prediction in the batch
            sample_names.append(name)

    # Save visualizations
    if input_samples and output_samples:
        save_visualizations(
            torch.stack(input_samples),
            torch.stack(output_samples),
            1000,
            1000,
            sample_names,
            visualization_dir,
        )

    df = pd.DataFrame({"ID": all_ids, "PL": all_pls})
    df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
