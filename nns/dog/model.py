import math
import numbers

import numpy as np
import torch
from scipy import spatial
from skimage.feature.blob import _blob_overlap
from torch import nn


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
        overlap=0.5,
    ):
        super().__init__()

        self.footprint = footprint
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
        self.threshold = nn.Parameter(torch.tensor(-threshold))
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        gaussian_images = self.gaussian_pyramid(input)
        # computing difference between two successive Gaussian blurred images
        # multiplying with standard deviation provides scale invariance
        dog_images = (gaussian_images[0][:-1] - gaussian_images[0][1:]) * (
            self.sigmas[:-1].unsqueeze(0).unsqueeze(0).T
        )

        local_maxima = self.max_pool(dog_images.unsqueeze(0)).squeeze(0)
        local_maxima = local_maxima + self.threshold
        local_maxima = self.relu(local_maxima)
        # mask = local_maxima == (dog_images + self.threshold)
        soft_mask = 1 - (local_maxima - (dog_images + self.threshold))
        return local_maxima, soft_mask

    def make_blobs(self, mask, local_maxima=None):
        if local_maxima is not None:
            local_maxima = local_maxima[mask].cpu().numpy()
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


def prune_blobs(blobs_array, overlap, local_maxima=None, *, sigma_dim=1):
    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array

    for (i, j) in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        blob_overlap = _blob_overlap(blob1, blob2, sigma_dim=sigma_dim)
        if blob_overlap > overlap:
            if local_maxima is not None:
                if local_maxima[i] > local_maxima[j]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
            else:
                blob2[-1] = (blob1[-1] + blob2[-1]) / 2
                blob1[-1] = 0

    return blobs_array[blobs_array[:, -1] > 0]
