import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt

def bresenham2D(x0, y0, x1, y1):
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

def generate_los_map_from_path(rgb_path):
    rgb_path = Path(rgb_path)
    fname = rgb_path.stem  # e.g., B1_Ant1_f1_S0

    # 解析出发射器 index，例如 S0 => 0
    s_idx = int(fname.split("_")[-1][1:])  # 获取最后一个"Sx"，并提取数字

    # 构建 position 文件路径
    building = fname.split("_")[0]      # e.g., B1
    antenna = fname.split("_")[1]       # e.g., Ant1
    freq = fname.split("_")[2]          # e.g., f1
    pos_file = Path(f"./Positions/Positions_{building}_{antenna}_{freq}.csv")

    # 加载发射器坐标
    df = pd.read_csv(pos_file)
    tx_x = int(round(df.iloc[s_idx]['X']))
    tx_y = int(round(df.iloc[s_idx]['Y']))

    # 加载 RGB 图像并转 tensor
    to_tensor = transforms.ToTensor()
    rgb_tensor = to_tensor(Image.open(rgb_path).convert("RGB"))
    R = rgb_tensor[0].numpy()
    G = rgb_tensor[1].numpy()
    H, W = R.shape

    wall_mask = ((R != 0) | (G != 0)).astype(np.uint8)
    los_map = np.ones((H, W), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            path = bresenham2D(tx_x, tx_y, x, y)
            for px, py in path:
                if 0 <= px < W and 0 <= py < H:
                    if wall_mask[py, px]:
                        los_map[y, x] = 0
                        break

    return los_map

# 示例
rgb_path = "./inputs/B1_Ant1_f1_S48.png"
los = generate_los_map_from_path(rgb_path)

# 可视化
plt.imshow(los, cmap='gray')
plt.title("LOS Map")
plt.axis('off')
plt.show()

# 保存为 .npy 和 .png
np.save("./los_maps/B1_Ant1_f1_S48.npy", los)
plt.imsave("./los_maps/B1_Ant1_f1_S48.png", los, cmap='gray')
