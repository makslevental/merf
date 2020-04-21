import glob
from pathlib import Path

import numpy as np
import torch
from skimage import io, img_as_float
from skimage.filters import gaussian
from torch.utils.data import Dataset

from sk_image.enhance_contrast import stretch_composite_histogram


class Trivial(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_path, num_repeats=10):
        self.img_path = img_path
        self.repeats = num_repeats

    def transform(self, img):
        img = img_as_float(img)
        img = gaussian(img, sigma=1)
        img = stretch_composite_histogram(img)
        img = torch.from_numpy(img[np.newaxis, :, :]).float()
        return img

    def __len__(self):
        return self.repeats

    def __getitem__(self, idx):
        image = io.imread(self.img_path, as_gray=True)
        t_img = self.transform(image)
        return t_img


class PLIF(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, plif_dir: Path):
        self.img_paths = glob.glob(str(plif_dir / "*.TIF"))

    def transform(self, img):
        img = img_as_float(img)
        img = gaussian(img, sigma=1)
        img = stretch_composite_histogram(img)
        img = torch.from_numpy(img[np.newaxis, :, :]).float()
        return img

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = io.imread(self.img_paths[idx], as_gray=True)
        t_img = self.transform(image)
        return t_img
