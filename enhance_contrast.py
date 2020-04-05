import glob
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from skimage import exposure
from skimage.io import imread

from blob import dog, make_circles_fig
from preprocess import make_figure, show, gamma

AUTO_THRESHOLD = 5000


@dataclass
class ImageStats:
    @dataclass
    class Histogram:
        bin_counts: np.ndarray
        bin_centers: np.ndarray

    pixel_count: int
    histogram: Histogram

    mean: float
    mode: float
    std_dev: float

    min: int
    max: int

    n_bins: int
    bin_width: float

    def __init__(self, image, n_hist_bins=256):
        self.n_bins = n_hist_bins

        px, py = image.shape
        self.pixel_count = px * py
        self.histogram = self.Histogram(*exposure.histogram(image, nbins=n_hist_bins))

        self.mean = image.mean()
        self.mode = stats.mode(image, axis=None).mode.item()
        self.std_dev = image.std()

        self.min = image.min()
        self.max = image.max()
        self.bin_width = (self.max - self.min) / self.n_bins


def get_min_and_max(stats: ImageStats, saturation=0.0):
    if saturation > 0:
        threshold = stats.pixel_count * saturation / 200
    else:
        threshold = 0

    count = 0
    for bin_count, bin_center in zip(
        stats.histogram.bin_counts, stats.histogram.bin_centers
    ):
        count += bin_count
        found = count > threshold
        if found:
            break

    h_min = bin_center

    count = 0
    for bin_count, bin_center in reversed(
        list(zip(stats.histogram.bin_counts, stats.histogram.bin_centers))
    ):
        count += bin_count
        found = count > threshold
        if found:
            break

    h_max = bin_center
    return h_min, h_max


def stretch_composite_histogram(image, stats: ImageStats, saturation=0.35):
    h_min, h_max = get_min_and_max(stats, saturation)
    return exposure.rescale_intensity(image, in_range=(h_min, h_max), out_range="image")


def test():
    # data_pth = Path("RawData")
    # image_fn = Path("R-233_5-8-6_000119.T000.D000.P000.H000.PLIF1.TIF")
    #
    # image_pth = data_pth / image_fn
    for image_pth in glob.glob("RawData/*.TIF"):
        img_org = imread(image_pth)
        start = time.time()
        g = gamma(img_org)
        blobs = dog(g)
        print("gamma", len(blobs))
        print(time.time() - start)

        start = time.time()
        stats = ImageStats(img_org)
        s = stretch_composite_histogram(img_org, stats)
        blobs = dog(s)
        print(time.time() - start)
        print("stretch", len(blobs))


if __name__ == "__main__":
    test()
