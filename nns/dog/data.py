import numpy as np
import torch
from skimage import io, img_as_float
from skimage.filters import gaussian
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Lambda

from sk_image.enhance_contrast import stretch_composite_histogram


class Trivial(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_path, num_repeats=10):
        self.img_path = img_path
        self.repeats = num_repeats
        self.tfms = transforms.Compose([
            Lambda(lambda img: img_as_float(img)),
            Lambda(lambda img: gaussian(img, sigma=1)),
            Lambda(lambda img: stretch_composite_histogram(img)),
            Lambda(lambda img: torch.from_numpy(img[np.newaxis, :, :]).float()),
        ])

    def __len__(self):
        return self.repeats

    def __getitem__(self, _idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        image = io.imread(self.img_path, as_gray=True)
        t_img = self.tfms(image)
        return t_img
