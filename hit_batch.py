import cupy as cp
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

def _bresenhamline_nslope(slope):
    scale = cp.amax(cp.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = 1
    normalizedslope = slope / scale
    normalizedslope[zeroslope] = 0
    return normalizedslope

def _bresenhamlines(start, end, max_iter=-1):
    if max_iter == -1:
        max_iter = cp.amax(cp.abs(end - start), axis=1)
    max_iter = max_iter.astype(cp.int32).max()
    steps = cp.arange(1, max_iter + 1, dtype=cp.int32).reshape(-1, 1)
    nslope = _bresenhamline_nslope(end - start)
    bline = start[:, None, :] + nslope[:, None, :] * steps
    return cp.rint(bline).astype(cp.int32)

def generate_wall_mask(png_path):
    img = Image.open(png_path).convert("RGB")
    rgb_tensor = transforms.ToTensor()(img)
    R = rgb_tensor[0].numpy()
    G = rgb_tensor[1].numpy()
    return ((R != 0) | (G != 0)).astype(np.uint8)

def generate_hit_map(wall_mask, tx_x, tx_y, batch_size=70000):
    H, W = wall_mask.shape
    wall_gpu = cp.asarray(wall_mask, dtype=cp.uint8)

    y, x = cp.meshgrid(cp.arange(H), cp.arange(W), indexing='ij')
    all_points = cp.stack((y.ravel(), x.ravel()), axis=1).astype(cp.int32)

    tx = cp.array([[tx_x, tx_y]], dtype=cp.int32)
    hit_map = cp.zeros(H * W, dtype=cp.int32)

    for i in range(0, len(all_points), batch_size):
        batch_points = all_points[i:i + batch_size]
        tx_batch = cp.repeat(tx, batch_points.shape[0], axis=0)

        lines = _bresenhamlines(batch_points, tx_batch)
        flat_lines = lines.reshape(-1, 2)

        valid_mask = (
            (flat_lines[:, 0] >= 0) & (flat_lines[:, 0] < H) &
            (flat_lines[:, 1] >= 0) & (flat_lines[:, 1] < W)
        )
        wall_values = cp.zeros(flat_lines.shape[0], dtype=cp.uint8)
        valid_lines = flat_lines[valid_mask]
        wall_values[valid_mask] = wall_gpu[valid_lines[:, 0], valid_lines[:, 1]]
        wall_values = wall_values.reshape(lines.shape[0], lines.shape[1])

        diff = cp.diff(wall_values, axis=1)
        hits = cp.sum(diff == 1, axis=1)
        hit_map[i:i + hits.shape[0]] = hits

    return cp.asnumpy(hit_map.reshape(H, W))

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
            tx_x = int(df.loc[s_idx, "X"]) - 1
            tx_y = int(df.loc[s_idx, "Y"]) - 1

            wall_mask = generate_wall_mask(img_path)
            hit_map = generate_hit_map(wall_mask, tx_x, tx_y)

            np.save(output_dir / f"{scene}_S{s_idx}_hit.npy", hit_map)

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

if __name__ == "__main__":
    process_all("inputs", "Positions", "hitmap")
