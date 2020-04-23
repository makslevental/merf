import glob
import math
import numbers
import os
from os.path import abspath
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from scipy import spatial
from skimage import img_as_float
from skimage.feature.blob import _blob_overlap
from skimage.filters import gaussian
from skimage.io import imread
from torch import nn

from nns.dog.data import Trivial
from sk_image.blob import make_circles_fig
from sk_image.enhance_contrast import stretch_composite_histogram
from sk_image.preprocess import make_figure

DATA_DIR = os.environ.get("FSP_DATA_DIR")
if DATA_DIR is None:
    raise Exception("need to specify env var FSP_DATA_DIR")
DATA_DIR = Path(abspath(DATA_DIR))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prune_blobs(blobs_array, overlap, local_maxima, *, sigma_dim=1):
    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array

    for (i, j) in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        blob_overlap = _blob_overlap(blob1, blob2, sigma_dim=sigma_dim)
        if blob_overlap == 1.0:
            if local_maxima[i] > local_maxima[j]:
                blob2[-1] = 0
            else:
                blob1[-1] = 0
        elif overlap < blob_overlap < 1.0:
            if blob1[-1] < blob2[-1]:
                blob2[-1] = 0
            else:
                blob1[-1] = 0

    return blobs_array[blobs_array[:, -1] > 0]


def torch_gaussian_kernel(width=21, sigma=3, dim=2):
    if isinstance(width, numbers.Number):
        width = [width] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in width]
    )
    for size, std, mgrid in zip(width, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
            1
            / (std * math.sqrt(2 * math.pi))
            * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)
    return kernel


class DifferenceOfGaussians(nn.Module):
    def __init__(
        self,
        *,
        max_sigma=10,
        min_sigma=1,
        sigma_bins=50,
        truncate=5.0,
        footprint=3,
        threshold=0.001,
        prune=True,
        overlap=0.5
    ):
        super().__init__()

        self.footprint = footprint
        self.threshold = threshold
        self.prune = prune
        self.overlap = overlap

        self.sigma_list = np.linspace(
            start=min_sigma,
            stop=max_sigma + (max_sigma - min_sigma) / sigma_bins,
            num=sigma_bins + 1,
        )
        sigmas = torch.from_numpy(self.sigma_list)
        self.register_buffer("sigmas", sigmas)
        print("gaussian pyramid sigmas: ", len(sigmas), sigmas)

        # max is performed in order to accommodate largest filter
        self.max_radius = int(truncate * max(sigmas) + 0.5)
        self.gaussian_pyramid = nn.Conv2d(
            1,  # greyscale input
            sigma_bins + 1,  # sigma+1 filters so that there are sigma dogs
            2 * self.max_radius
            + 1,  # conv stack should be as wide as widest gaussian filter
            bias=False,
            padding=self.max_radius,  # hence no shrink of image
            padding_mode="zeros",
        )

        for i, s in enumerate(sigmas):
            radius = int(truncate * s + 0.5)
            kernel = torch_gaussian_kernel(width=2 * radius + 1, sigma=s.item())
            pad_size = self.max_radius - radius
            if pad_size > 0:
                padded_kernel = nn.ConstantPad2d(pad_size, 0)(kernel)
            else:
                padded_kernel = kernel
            self.gaussian_pyramid.weight.data[i].copy_(padded_kernel)

        self.padding = (self.footprint - 1) // 2
        if not isinstance(self.footprint, int):
            self.footprint = tuple(self.footprint)
            self.padding = tuple(self.padding)

        self.max_pool = nn.MaxPool3d(
            kernel_size=self.footprint, padding=self.padding, stride=1
        )
        self.max_conv = nn.Conv3d(
            1,1,kernel_size=self.footprint, padding=self.padding, stride=1
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        gaussian_images = self.gaussian_pyramid(input)
        # computing difference between two successive Gaussian blurred images
        # multiplying with standard deviation provides scale invariance
        dog_images = (gaussian_images[0][:-1] - gaussian_images[0][1:]) * (
            self.sigmas[:-1].unsqueeze(0).unsqueeze(0).T
        )
        image_max = self.max_pool(dog_images.unsqueeze(0)).squeeze(0)
        mask = dog_images == image_max
        mask &= dog_images > self.threshold

        local_maxima = image_max[mask].cpu().numpy()
        coords = mask.nonzero().cpu().numpy()
        cds = coords.astype(np.float64)

        # translate final column of cds, which contains the index of the
        # sigma that produced the maximum intensity value, into the sigma
        sigmas_of_peaks = self.sigma_list[coords[:, 0]]
        # Remove sigma index and replace with sigmas
        cds = np.hstack([cds[:, 1:], sigmas_of_peaks[np.newaxis].T])
        print("preprune blobs: ", len(cds))
        if self.prune:
            blobs = prune_blobs(cds, self.overlap, local_maxima, sigma_dim=1)
        else:
            blobs = cds

        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        return blobs


def torch_dog(dataloader, **dog_kwargs):
    with torch.no_grad():
        dog = DifferenceOfGaussians(**dog_kwargs).to(DEVICE)
        for p in dog.parameters():
            p.requires_grad = False
        dog.eval()
        for img_tensor in dataloader:
            img_tensor = img_tensor.to(DEVICE)
            blobs = dog(img_tensor)
    return blobs


def torch_dog_test():
    with torch.no_grad():
        input = torch.rand(1, 1, 1000, 1000)
        input[0, 0, 400:600, 400:600] = 1
        input.to(DEVICE)
        make_figure(input.detach().numpy()[0][0]).show()
        dogs = torch_dog(input)
        for d in dogs:
            make_figure(d.detach().numpy()).show()


def torch_dog_img_test():
    # this is a stupid hack because running remote and profiling messes with file paths
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../../simulation/screenshot.png"
    )
    screenshot = Trivial(img_path=image_pth, num_repeats=1)
    make_figure(screenshot[0].squeeze(0).numpy()).show()
    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=True
    )

    blobs = torch_dog(
        train_dataloader,
        min_sigma=1,
        max_sigma=40,
        prune=True,
        overlap=0.9,
        threshold=0.1,
    )
    print(len(blobs))
    make_circles_fig(screenshot[0].squeeze(0).numpy(), blobs).show()
    plt.hist([r for (_, _, r) in blobs], bins=256)
    plt.show()


def main():
    for image_pth in glob.glob(DATA_DIR / "*.TIF"):
        img_orig = imread(image_pth, as_gray=True)
        # values have to be float and also between 0,1 for peak finding to work
        img_orig = img_as_float(img_orig)
        filtered_img = gaussian(img_orig, sigma=1)
        s2 = stretch_composite_histogram(filtered_img)
        t_image = torch.from_numpy(s2[np.newaxis, np.newaxis, :, :]).float()
        t_image = t_image.to(DEVICE)
        blobs = torch_dog(t_image, prune=True)
        print("blobs: ", len(blobs))
        break


def test_gaussian_kernel():
    t = torch_gaussian_kernel()
    X = np.arange(1, 22)
    Y = np.arange(1, 22)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        X, Y, t.numpy(), cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == "__main__":
    # torch_dog_test()
    # a = torch.zeros(10, dtype=torch.bool)
    # print(a.int())
    torch_dog_img_test()
    # main()
