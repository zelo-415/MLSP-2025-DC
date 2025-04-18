import cupy as cp
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def _bresenhamline_nslope(slope):
    scale = cp.amax(cp.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = cp.ones(1)
    normalizedslope = cp.array(slope, dtype=cp.double) / scale
    normalizedslope[zeroslope] = cp.zeros(slope[0].shape)
    return normalizedslope

def _bresenhamlines(start, end, max_iter):
    if max_iter == -1:
        max_iter = cp.amax(cp.amax(cp.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)
    steps = cp.arange(1, max_iter + 1)
    stepmat = cp.tile(steps, (dim, 1)).T
    bline = start[:, cp.newaxis, :] + nslope[:, cp.newaxis, :] * stepmat
    return cp.rint(bline).astype(cp.int32)

def generate_los_map(rgb_path, positions_dir, output_dir):
    fname = rgb_path.stem
    s_idx = int(fname.split("_")[-1][1:])
    building, antenna, freq = fname.split("_")[:3]
    pos_file = Path(positions_dir) / f"Positions_{building}_{antenna}_{freq}.csv"
    df = pd.read_csv(pos_file)
    tx_x = int(round(df.iloc[s_idx]['X']))
    tx_y = int(round(df.iloc[s_idx]['Y']))

    rgb_tensor = transforms.ToTensor()(Image.open(rgb_path).convert("RGB"))
    R, G = rgb_tensor[0].numpy(), rgb_tensor[1].numpy()
    wall_mask = ((R != 0) | (G != 0)).astype(np.uint8)
    H, W = wall_mask.shape

    wall_cp = cp.asarray(wall_mask)
    y, x = cp.meshgrid(cp.arange(H), cp.arange(W), indexing='ij')
    ground_points = cp.stack([x.ravel(), y.ravel()], axis=1)

    tx = cp.array([[tx_x, tx_y]])
    lines = _bresenhamlines(ground_points, tx.repeat(len(ground_points), axis=0), max_iter=-1)
    flat_lines = lines.reshape(-1, 2)
    hits = wall_cp[flat_lines[:, 1], flat_lines[:, 0]].reshape(lines.shape[0], lines.shape[1])
    blocked = cp.any(hits == 1, axis=1)
    los_map = (~blocked).reshape(H, W).astype(cp.uint8)

    np.save(Path(output_dir) / f"{fname}.npy", cp.asnumpy(los_map))

def process_all_images(inputs_dir, positions_dir, output_dir):
    inputs_dir = Path(inputs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    for rgb_path in tqdm(sorted(inputs_dir.glob("*.png")), desc="Generating LOS (CuPy)"):
        try:
            generate_los_map(rgb_path, positions_dir, output_dir)
        except Exception as e:
            print(f"[ERROR] {rgb_path.name}: {e}")

if __name__ == "__main__":
    process_all_images("inputs", "Positions", "los_map")
