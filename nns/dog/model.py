import math
import numbers
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import spatial
from skimage import img_as_float, io
from skimage.feature.blob import _blob_overlap
from torch import nn

from sk_image.blob import make_circles_fig
from sk_image.preprocess import make_figure


class DifferenceOfGaussians(nn.Module):
    def __init__(
        self,
        img_height,
        img_width,
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
        super(DifferenceOfGaussians, self).__init__()
        self.footprint = footprint
        self.prune = prune
        self.overlap = overlap
        self.threshold = threshold
        self.signal_ndim = 2

        self.sigma_list = np.linspace(
            start=min_sigma,
            stop=max_sigma + (max_sigma - min_sigma) / sigma_bins,
            num=sigma_bins + 1,
        )
        sigmas = torch.from_numpy(self.sigma_list)
        self.register_buffer("sigmas", sigmas)
        print("gaussian pyramid sigmas: ", len(sigmas), sigmas)

        # max is performed in order to accommodate largest filter
        max_radius = int(truncate * max(sigmas) + 0.5)
        max_width = 2 * max_radius + 1
        # Compute convolution using fft.
        padded_size_1 = img_height + max_width - 1
        padded_size_2 = img_width + max_width - 1
        # Round up to next power of 2 for cheaper fft.
        self.fast_ftt_size_1 = 2 ** math.ceil(math.log2(padded_size_1))
        self.fast_ftt_size_2 = 2 ** math.ceil(math.log2(padded_size_2))
        self.f_gaussian_pyramid = []
        self.kernel_paddings = []
        for i, s in enumerate(sigmas):
            radius = int(truncate * s + 0.5)
            width = 2 * radius + 1
            kernel = torch_gaussian_kernel(width=width, sigma=s.item())
            self.kernel_paddings.append(radius)
            padded_kernel = F.pad(
                kernel,
                # pad last dim by (0, fast_ftt_size_2 - m2) and 2nd to last by ...
                (0, self.fast_ftt_size_2 - width, 0, self.fast_ftt_size_1 - width),
            )
            f_kernel = torch.rfft(
                padded_kernel, signal_ndim=self.signal_ndim, onesided=True
            )
            self.f_gaussian_pyramid.append(f_kernel)
        self.f_gaussian_pyramid = torch.stack(self.f_gaussian_pyramid, dim=0)

        self.padding = (self.footprint - 1) // 2
        if not isinstance(self.footprint, int):
            self.footprint = tuple(self.footprint)
            self.padding = tuple(self.padding)

        self.max_pool = nn.MaxPool3d(
            kernel_size=self.footprint, padding=self.padding, stride=1
        )

    def forward(self, input: torch.Tensor):
        m1, m2 = list(input.size())[-self.signal_ndim :]
        padded_input = F.pad(
            input,
            # pad last dim by (0, fast_ftt_size_2 - m2) and 2nd to last by ...
            (0, self.fast_ftt_size_2 - m2, 0, self.fast_ftt_size_1 - m1),
        )
        f_input = torch.rfft(padded_input, signal_ndim=self.signal_ndim, onesided=True)

        f_gaussian_images = comp_mul(self.f_gaussian_pyramid, f_input)
        gaussian_images = torch.irfft(
            f_gaussian_images,
            signal_ndim=self.signal_ndim,
            onesided=True,
            signal_sizes=padded_input.shape[1:],
        )
        gs = torch.zeros(*(gaussian_images.shape[:-self.signal_ndim]+input.shape[-self.signal_ndim:]))
        for i, radius in enumerate(self.kernel_paddings):
            gs[:, i] = gaussian_images[:, i, radius : m1 + radius, radius : m2 + radius]
        gaussian_images = gs

        # computing difference between two successive Gaussian blurred images
        # multiplying with standard deviation provides scale invariance
        dog_images = (gaussian_images[:, :-1] - gaussian_images[:, 1:]) * (
            self.sigmas[:-1].unsqueeze(0).unsqueeze(0).T
        )
        local_maxima = self.max_pool(dog_images)
        mask = (local_maxima == dog_images) & (dog_images > self.threshold)
        return mask, local_maxima

    def make_blobs(self, mask, local_maxima=None):
        if local_maxima is not None:
            local_maxima = local_maxima[mask].detach().cpu().numpy()
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
            * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
        )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)
    return kernel


def comp_mul(a, b):
    op = partial(torch.einsum, "fxy,bxy->bfxy")
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return torch.stack([op(ar, br) - op(ai, bi), op(ar, bi) + op(ai, br)], dim=-1)


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


def test():
    img = img_as_float(
        io.imread(
            "/Users/maksim/dev_projects/merf/simulation/screenshot.png", as_gray=True
        )
    )
    img_height, img_width = img.shape
    plt.imshow(img)
    plt.show()
    img = torch.from_numpy(img)
    imgs = torch.stack([img, img, img], dim=0)
    # imgs = torch.ones_like(imgs)
    dog = DifferenceOfGaussians(img_height, img_width, sigma_bins=20, threshold=0.1)
    masks, local_maximas = dog(imgs)
    print(len(masks))
    for m, l in zip(masks, local_maximas):
        blobs = dog.make_blobs(m, l)
        make_circles_fig(img.numpy(), blobs).show()


if __name__ == "__main__":
    test()
