import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from sk_image.preprocess import make_figure


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, *, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class DifferenceOfGaussians(nn.Module):
    def __init__(self, *, max_sigma=10, min_sigma=1, threshold=0.005, overlap=0.8, sigma_ratio=1.6):
        super().__init__()

        self.threshold = threshold
        self.overlap = overlap

        # k such that min_sigma*(sigma_ratio**k) > max_sigma
        k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))
        # a geometric progression of standard deviations for gaussian kernels
        self.sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                               for i in range(k + 1)])


        self.gaussian_pyramid = None


    def forward(self, input: torch.Tensor) -> torch.Tensor:

        gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

        # computing difference between two successive Gaussian blurred images
        # multiplying with average standard deviation provides scale invariance
        dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                      * np.mean(sigma_list[i]) for i in range(k)]

        image_cube = np.stack(dog_images, axis=-1)


def test():
    smoothing = GaussianSmoothing(channels=1, kernel_size=5, sigma=1)
    input = torch.rand(1, 1, 100, 100)
    make_figure(input.detach().numpy()[0][0]).show()
    input = F.pad(input, (2, 2, 2, 2), mode='reflect')
    output = smoothing(input)
    make_figure(output.detach().numpy()[0][0]).show()


if __name__ == '__main__':
    test()
