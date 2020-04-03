import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from scipy.stats import median_absolute_deviation as mad
from skimage import exposure, img_as_float
from skimage.color import rgb2gray
from skimage.filters import try_all_threshold
from skimage.io import imread

# match_img = rgb2gray(imread("RawData" / Path("R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.jpeg")))


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


def make_hist(img, n_bins=1024):
    fig, ax = plt.subplots(tight_layout=True)
    img = img[img > 0]
    ax.hist(img, bins=n_bins, density=True)

    return fig


def gamma(img, g=10):
    return exposure.adjust_gamma(img, 1 / g)


def clahe(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.5, kernel_size=10)
    return img_adapteq


def show(img):
    make_figure(img).show()
    make_hist(img).show()


def stretch(img):
    img = img_as_float(img)
    p2, p98 = numpy.percentile(img, (10, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


def equalization(img):
    return exposure.equalize_hist(img)


def log_transform(img):
    return exposure.adjust_log(img)


# def match(img, reference=match_img):
#     return exposure.match_histograms(img, reference)


def threshold(img):
    # thresh = threshold_otsu(img)
    # binary = numpy.copy(img)
    # binary[binary > thresh] = 1
    # binary[binary < thresh] = 0
    # return binary
    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.show()


if __name__ == "__main__":
    # data_pth = Path("RawData")
    # image_fn = Path("R-233_5-8-6_000114.T000.D000.P000.H000.PLIF1.TIF")
    # image_pth = data_pth / image_fn
    # img_org = imread(image_pth)
    # show(img_org)
    for image_fn in glob.glob("RawData/*.TIF"):
        img_org = imread(image_fn)
        # s = stretch(img_org)
        # show(s)
        g = gamma(img_org)
        show(g)

    # c = clahe(img_org)
    # show(c)
    #
    # t = threshold(g)
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
