import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def generate_wall_mask(png_path):
    img = Image.open(png_path).convert('RGB')
    img_np = np.array(img)
    R, G = img_np[:, :, 0], img_np[:, :, 1]
    return ((R != 0) | (G != 0)).astype(np.uint8)

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    sx, sy = 1 if x1 >= x0 else -1, 1 if y1 >= y0 else -1
    if dx >= dy:
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        points.append((x1, y1))
    else:
        err = dy // 2
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        points.append((x1, y1))
    return points

def generate_los_mask(wall_mask, tx_x, tx_y):
    H, W = wall_mask.shape
    los_mask = np.zeros_like(wall_mask, dtype=np.uint8)
    for x in range(H):
        for y in range(W):
            path = bresenham_line(tx_x, tx_y, x, y)
            blocked = any(wall_mask[px, py] == 1 for (px, py) in path if (px, py) != (tx_x, tx_y))
            los_mask[x, y] = 0 if blocked else 1
    return los_mask

def extract_scene_and_index(filename):

    name = Path(filename).stem
    parts = name.split('_')
    scene = '_'.join(parts[:-1])
    index = int(parts[-1][1:])  
    return scene, index

def process_all_inputs(input_dir, csv_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_pngs = sorted(f for f in os.listdir(input_dir) if f.endswith(".png"))

    for fname in tqdm(all_pngs, desc="Processing Images"):
        scene_name, tx_index = extract_scene_and_index(fname)
        png_path = os.path.join(input_dir, fname)
        csv_path = os.path.join(csv_dir, f"Positions_{scene_name}.csv")


        pos_df = pd.read_csv(csv_path)
     

        tx_x, tx_y = int(pos_df.loc[tx_index, "X"]), int(pos_df.loc[tx_index, "Y"])
        wall_mask = generate_wall_mask(png_path)
        los_mask = generate_los_mask(wall_mask, tx_x, tx_y)

        out_path = os.path.join(output_dir, f"{scene_name}_S{tx_index}_los.npy")
        np.save(out_path, los_mask)

if __name__ == "__main__":
    process_all_inputs(
        input_dir="./inputs",
        csv_dir="./Positions",
        output_dir="./losmap"
    )
