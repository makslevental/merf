from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from scipy.stats import median_absolute_deviation as mad
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.io import imread


def mad_normalize(x):
    med = numpy.median(x)
    x_mad = mad(x.flatten())
    return (x - med) / x_mad


def make_figure(im, dpi=96) -> plt.Figure:
    px, py = im.shape
    fig = plt.figure(figsize=(py / numpy.float(dpi), px / numpy.float(dpi)))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], yticks=[], xticks=[], frame_on=False)
    ax.imshow(im, cmap="gray")

    return fig


def make_hist(img, n_bins=256):
    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(img.ravel(), bins=n_bins)

    return fig


def gamma(img, g=5):
    return exposure.adjust_gamma(img, 1 / g)


def clahe(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.05)
    return img_adapteq


def show(img):
    make_figure(img).show()
    make_hist(img).show()


def stretch(img):
    p2, p98 = numpy.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


def equalization(img):
    return exposure.equalize_hist(img)


def log_transform(img):
    return exposure.adjust_log(img)


def threshold(img):
    thresh = threshold_otsu(img)
    binary = numpy.copy(img)
    binary[binary > thresh] = 1
    binary[binary < thresh] = 0
    return binary


if __name__ == "__main__":
    data_pth = Path("RawData/")
    image_fn = Path("R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.TIF")
    image_pth = data_pth / image_fn

    img_org = imread(image_pth)
    show(img_org)

    g = gamma(img_org)
    show(g)

    # c = clahe(img_org)
    # show(c)
    #
    # t = threshold(c)
    # show(t)
    #
    # s = stretch(img_org)
    # show(s)
    #
    # e = equalization(img_org)
    # show(e)
    #
    # l = log_transform(img_org)
    # show(l)
