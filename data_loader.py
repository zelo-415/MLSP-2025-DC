import os
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
import cv2
import torch


def load_sampling_mask(mask_shape):
    """
    Create a random sparse sampling mask (e.g. 0.5% of pixels).
    """
    mask = np.zeros(mask_shape)
    num_points = int(0.005 * mask_shape[0] * mask_shape[1])  # Example: 0.5%
    idx = np.random.choice(mask.size, num_points, replace=False)
    np.put(mask, idx, 1)
    return mask.astype(np.uint8)

def padding(image, target_size, pad_value=0):
    """
    Pad the image to the target size.
    """
    h, w = image.shape[:2]
    pad_h = max(0, target_size[0] - h)
    pad_w = max(0, target_size[1] - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if image.ndim == 2:
        return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=pad_value)
    elif image.ndim == 3:
        return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=pad_value)

class PLDataset(Dataset):
    def __init__(self, file_names, input_path, output_path, resize=(512, 512), use_sparse=True):
        self.file_names = file_names
        self.input_path = input_path
        self.output_path = output_path
        self.resize = resize
        self.use_sparse = use_sparse
        self.pl_max = 160.0
        self.pl_min = 13.0

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        X = imread(os.path.join(self.input_path, fname + ".png"))
        y = imread(os.path.join(self.output_path, fname + ".png"))

        y = y.astype(np.float32)
        X = X.astype(np.float32)

        mask = load_sampling_mask(y.shape) if self.use_sparse else np.ones_like(y)
        sparse_y = y * mask
        sparse_y = sparse_y.astype(np.float32)
        mask = mask.astype(np.float32)

        # Normalize y
        y = (y - self.pl_min) / (self.pl_max - self.pl_min)
        sparse_y = (sparse_y - self.pl_min) / (self.pl_max - self.pl_min)

        # Resize images
        target_size = self.resize
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        X = padding(X, target_size)
        y = padding(y, target_size)
        sparse_y = padding(sparse_y, target_size)
        mask = padding(mask, target_size)

        X = np.concatenate([X, sparse_y[..., None], mask[..., None]], axis=-1)
        assert X.shape[:2] == (self.resize[0], self.resize[1]), f"Padding shape mismatch: {X.shape[:2]} vs {self.resize}"

        X = np.moveaxis(X, -1, 0)  
        y = np.expand_dims(y, axis=0)
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(mask)


def create_dataloader(file_names, input_path, output_path, batch_size=4, shuffle=True):
    dataset = PLDataset(file_names, input_path, output_path)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)