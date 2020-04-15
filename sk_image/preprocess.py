from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from scipy.stats import median_absolute_deviation as mad
from skimage import exposure
from skimage.filters import sobel, threshold_otsu
from skimage.filters.rank import majority
from skimage.io import imread
from skimage.morphology import disk, opening, closing


def height(image):
    return sobel(image)


def mad_normalize(x):
    med = numpy.median(x)
    x_mad = mad(x.flatten())
    return (x - med) / x_mad


def make_figure(im, title=None, dpi=96, norm=None) -> plt.Figure:
    px, py = im.shape
    fig = plt.figure(figsize=(py / numpy.float(dpi), px / numpy.float(dpi)))
    if title is None:
        dims = [0.0, 0.0, 1.0, 1.0]
    else:
        dims = [0.0, 0.0, 1.0, 0.95]
    ax = fig.add_axes(dims, yticks=[], xticks=[], frame_on=False)
    ax.imshow(im, cmap="gray", norm=norm)
    ax.set_title(title, fontsize=50)

    return fig


def make_hist(img, title=None, use_log_scale=False, n_bins=256):
    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(img.ravel(), bins=n_bins)
    ax.set_title(title)
    if use_log_scale:
        ax.set_yscale("log")
    return fig


def gamma(img, g=10):
    return exposure.adjust_gamma(img, 1 / g)


def clahe(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.5, kernel_size=10)
    return img_adapteq


def show(img, title=None, use_log_hist=False):
    make_figure(img, title).show()
    make_hist(img, title, use_log_scale=use_log_hist).show()


def major(img):
    maj_img = majority(img, disk(5))
    return maj_img


def stretch(img, saturation=0.35):
    p2, p98 = numpy.percentile(img, (saturation / 100, 100 - (saturation / 100)))
    return exposure.rescale_intensity(img, in_range=(p2, p98), out_range="image")


def equalization(img):
    return exposure.equalize_hist(img)


def log_transform(img):
    return exposure.adjust_log(img)


def morphological_filter(img):
    thresh = threshold_otsu(img)
    print(thresh, len(img[img >= thresh].ravel()) / len(img.ravel()))
    binary = img >= thresh
    make_figure(binary).show()
    o_binary = opening(binary)
    make_figure(o_binary).show()
    c_binary = closing(o_binary)
    make_figure(c_binary).show()


if __name__ == "__main__":
    data_pth = Path("RawData")
    image_fn = Path("R-233_5-8-6_000119.T000.D000.P000.H000.PLIF1.TIF")
    image_pth = data_pth / image_fn
    img_org = imread(image_pth)
    show(img_org)
    s = stretch(img_org)
    show(s)
