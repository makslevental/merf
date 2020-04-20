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
from skimage import img_as_float
from skimage.feature.blob import _prune_blobs
from skimage.filters import gaussian
from skimage.io import imread
from torch import nn

from sk_image.blob import make_circles_fig
from sk_image.enhance_contrast import stretch_composite_histogram
from sk_image.preprocess import make_figure

DATA_DIR = os.environ.get("FSP_DATA_DIR")
if DATA_DIR is None:
    raise Exception("need to specify env var FSP_DATA_DIR")
DATA_DIR = Path(abspath(DATA_DIR))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        sigma_ratio=1.2,
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

        # k such that min_sigma*(sigma_ratio**k) > max_sigma
        self.k = int(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1)
        # a geometric progression of standard deviations for gaussian kernels
        self.sigma_list = np.asarray(
            [min_sigma * (sigma_ratio ** i) for i in range(self.k + 1)]
        )
        sigmas = torch.from_numpy(self.sigma_list)
        self.register_buffer("sigmas", sigmas)
        print("gaussian pyramid sigmas: ", len(sigmas))
        # max is performed in order to accommodate largest filter
        self.max_radius = int(truncate * max(sigmas) + 0.5)
        self.gaussian_pyramid = nn.Conv2d(
            1,
            self.k + 1,
            2 * self.max_radius + 1,
            bias=False,
            padding=self.max_radius,
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        gaussian_images = self.gaussian_pyramid(input)
        # computing difference between two successive Gaussian blurred images
        # multiplying with standard deviation provides scale invariance
        dog_images = (gaussian_images[0][:-1] - gaussian_images[0][1:]) * (
            self.sigmas[: self.k].unsqueeze(0).unsqueeze(0).T
        )
        image_max = self.max_pool(dog_images.unsqueeze(0))

        mask = dog_images == image_max.squeeze(0)
        mask &= dog_images > self.threshold

        # np.nonzero is faster than torch.nonzero()
        local_maxima = np.column_stack(mask.cpu().numpy().nonzero())
        lm = local_maxima.astype(np.float64)

        # translate final column of lm, which contains the index of the
        # sigma that produced the maximum intensity value, into the sigma
        sigmas_of_peaks = self.sigma_list[local_maxima[:, 0]]
        # Remove sigma index and replace with sigmas
        lm = np.hstack([lm[:, 1:], sigmas_of_peaks[np.newaxis].T])

        if self.prune:
            blobs = _prune_blobs(lm, self.overlap, sigma_dim=1)
        else:
            blobs = local_maxima

        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        return blobs


def torch_dog(
    img_tensor, min_sigma=1, max_sigma=10, sigma_ratio=1.01, prune=True, overlap=0.5
):
    with torch.no_grad():
        dog = DifferenceOfGaussians(
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            footprint=np.array((11, 3, 3)),
            prune=prune,
            overlap=overlap,
        ).to(DEVICE)
        for p in dog.parameters():
            p.requires_grad = False
        dog.eval()
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
    img_orig = imread(image_pth, as_gray=True)
    # values have to be float and also between 0,1 for peak finding to work
    img_orig = img_as_float(img_orig)
    filtered_img = gaussian(img_orig, sigma=1)
    s2 = stretch_composite_histogram(filtered_img)
    t_image = torch.from_numpy(s2).float().unsqueeze(0).unsqueeze(0)
    t_image = t_image.to(DEVICE)
    blobs = torch_dog(t_image, prune=True)
    print("blobs: ", len(blobs))
    # make_circles_fig(s2, blobs).show()


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
