import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
from pathlib import Path

def bresenham2D(x0, y0, x1, y1):
    """返回从 (x0, y0) 到 (x1, y1) 的 Bresenham route"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    points = []

    if dx > dy:
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
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

def generate_los_map_nopad(rgb_path, tx_x, tx_y):
    # 1. 加载图像并转换为 tensor
    to_tensor = transforms.ToTensor()
    rgb = Image.open(rgb_path).convert("RGB")
    rgb_tensor = to_tensor(rgb)  # shape [3, H, W]
    
    R = rgb_tensor[0].numpy()
    G = rgb_tensor[1].numpy()
    H, W = R.shape

    # 生成墙体 mask（1 表示有墙，0 表示无墙）
    wall_mask = ((R != 0) | (G != 0)).astype(np.uint8)

    # 初始化 los_map
    los_map = np.ones((H, W), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            path = bresenham2D(tx_x, tx_y, x, y)
            for px, py in path:
                if 0 <= px < W and 0 <= py < H:
                    if wall_mask[py, px]:  
                        los_map[y, x] = 0  # 被墙阻挡
                        break  

    return los_map  # 1 表示可达，0 表示不可达

rgb_path = "inputs/B1_Ant0_f1/S0.png"
pos_file = "Positions/Positions_B1_0.csv"

df = pd.read_csv(pos_file)
tx_x = int(round(df.iloc[0]['X']))
tx_y = int(round(df.iloc[0]['Y']))

los = generate_los_map_nopad(rgb_path, tx_x, tx_y)

# 保存为 .npy
np.save("los_map/B1_Ant0_f1.npy", los)
