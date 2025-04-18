import os
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
import cv2
import torch
import re
import math


def load_sampling_mask(mask_shape):
    """
    Create a random sparse sampling mask (e.g. 0.5% of pixels).
    """
    mask = np.zeros(mask_shape)
    num_points = int(0.005 * mask_shape[0] * mask_shape[1])  # Example: 0.5%
    idx = np.random.choice(mask.size, num_points, replace=False)
    np.put(mask, idx, 1)
    return mask.astype(np.uint8)


class PLDataset(Dataset):
    def __init__(self, file_names, input_path, output_path, resize=(512, 512), use_sparse=True, use_data_aug=False, 
            use_FSPL=True     ):
        self.file_names = file_names
        self.input_path = input_path
        self.output_path = output_path
        self.resize = resize
        self.use_sparse = use_sparse
        self.use_data_aug = use_data_aug
        self.use_FSPL = use_FSPL

    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        fname = self.file_names[idx]
        X = imread(os.path.join(self.input_path, fname + ".png"))
        y = imread(os.path.join(self.output_path, fname + ".png"))

        # Interpolation to resize
        X = cv2.resize(X, self.resize, interpolation=cv2.INTER_NEAREST)
        y = cv2.resize(y, self.resize, interpolation=cv2.INTER_CUBIC)

        fspl = self.find_FSPL(fname, X)
        #print(fspl.min(), fspl.max())

        y = y.astype(np.float32)
        X = X.astype(np.float32)

        mask = load_sampling_mask(y.shape) if self.use_sparse else np.ones_like(y)
        sparse_y = y * mask
        #print(fspl.shape)
        X = np.concatenate([X, sparse_y[..., None]], axis=-1)
        X = np.concatenate([X, fspl[..., None]], axis=-1)

        X = np.moveaxis(X, -1, 0)  
        y = np.expand_dims(y, axis=0)
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(mask)
    def find_FSPL(self, fname, X):
        freqs_GHz = [0.868, 1.8, 3.5]
        if self.use_FSPL:
            match = re.search(r'_f(\d+)', fname)
            fnum = int(match.group(1))
        freq_GHz = freqs_GHz[fnum-1]
        lam = 0.3 / freq_GHz
        fspl =  ( lam / (4 * math.pi * X[:, :, 2] + 1e-6) )**2
        fspl = 20 * np.log10 ( fspl )
        return fspl.astype(np.float32)



def create_dataloader(file_names, input_path, output_path, batch_size=4, shuffle=True):
    dataset = PLDataset(file_names, input_path, output_path)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)