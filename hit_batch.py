import cupy as cp
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

def _bresenhamlines_integer(start, end):
    """
    Vectorized integer Bresenham line generator using CuPy.
    Args:
        start: (N, 2) tensor, each row is (y1, x1)
        end: (N, 2) tensor, each row is (y2, x2)
    Returns:
        lines: (N, L, 2) tensor of integer coordinates
    """
    N = start.shape[0]
    dy = end[:, 0] - start[:, 0]
    dx = end[:, 1] - start[:, 1]

    steps = cp.maximum(cp.abs(dy), cp.abs(dx)) + 1  # shape (N,)
    max_len = int(cp.max(steps).item())  # convert to Python int to avoid ndarray error

    t = cp.arange(max_len, dtype=cp.int32).reshape(1, -1)  # (1, max_len)
    t = cp.broadcast_to(t, (N, max_len))

    ratio = cp.clip(t / (steps[:, None] - 1 + 1e-6), 0, 1)  # avoid division by zero

    y = cp.rint(start[:, 0:1] + ratio * dy[:, None]).astype(cp.int32)
    x = cp.rint(start[:, 1:2] + ratio * dx[:, None]).astype(cp.int32)

    lines = cp.stack((y, x), axis=-1)  # (N, max_len, 2)
    return lines

def generate_wall_mask(png_path):
    img = Image.open(png_path).convert("RGB")
    rgb_tensor = transforms.ToTensor()(img)
    R, G = rgb_tensor[0].numpy(), rgb_tensor[1].numpy()
    return ((R != 0) | (G != 0)).astype(np.uint8)

def generate_hit_map(wall_mask, tx_x, tx_y):
    H, W = wall_mask.shape
    wall_gpu = cp.asarray(wall_mask, dtype=cp.uint8)

    x, y = cp.meshgrid(cp.arange(H), cp.arange(W), indexing='ij')
    all_points = cp.stack((y.ravel(), x.ravel()), axis=1).astype(cp.int32)

    # Ensure tx_x, tx_y are plain Python ints
    tx_x = int(tx_x)
    tx_y = int(tx_y)
    tx = cp.array([[tx_y, tx_x]], dtype=cp.int32) 
    tx_batch = cp.repeat(tx, all_points.shape[0], axis=0)

    lines = _bresenhamlines_integer(all_points, tx_batch)  # (N, L, 2)
    N, L = lines.shape[:2]

    ys, xs = lines[..., 0], lines[..., 1]
    valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    flat_idx = ys * W + xs
    flat_idx[~valid] = 0  # out-of-bound locations set to 0 index

    flat_vals = wall_gpu.ravel()[flat_idx]
    flat_vals[~valid] = 0
    flat_vals = flat_vals.reshape(N, L)

    hits = cp.sum(flat_vals[:, 1:] == 1, axis=1)  # exclude TX point
    return cp.asnumpy(hits.reshape(H, W))

def process_all(inputs_dir, positions_dir, output_dir):
    inputs_dir = Path(inputs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images = sorted(inputs_dir.glob("*.png"))

    for img_path in tqdm(all_images, desc="Generating Hit Maps"):
        try:
            fname = img_path.stem
            scene, s_idx = '_'.join(fname.split('_')[:-1]), int(fname.split('_')[-1][1:])
            pos_path = Path(positions_dir) / f"Positions_{scene}.csv"
            df = pd.read_csv(pos_path)
            tx_x = int(df.loc[s_idx, "X"].item())
            tx_y = int(df.loc[s_idx, "Y"].item())

            wall_mask = generate_wall_mask(img_path)
            hit_map = generate_hit_map(wall_mask, tx_x, tx_y)
            np.save(output_dir / f"{scene}_S{s_idx}_hit.npy", hit_map)

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

if __name__ == "__main__":
    cp.cuda.Device(3).use()
    process_all("inputs", "Positions", "hitmap")
