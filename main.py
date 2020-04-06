import glob
import time
from pathlib import Path

from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.io import imread

from blob import dog, make_circles_fig, kornia_hessian, log, doh
from enhance_contrast import ImageStats, stretch_composite_histogram
from preprocess import gamma, make_figure, show

DEBUG = False


def count_droplets_gamma(img):
    img = gamma(img)
    blobs = dog(img)
    if DEBUG:
        make_circles_fig(img, blobs).show()
    return len(blobs)


def count_droplets_dog(img):
    stats = ImageStats(img)
    s = stretch_composite_histogram(img, stats)
    show(s)
    blobs = dog(s)
    if DEBUG:
        make_circles_fig(s, blobs, title=f"dog {len(blobs)}").show()
    return len(blobs)


def count_droplets_log(img):
    stats = ImageStats(img)
    s = stretch_composite_histogram(img, stats)
    blobs = log(s)
    if DEBUG:
        make_circles_fig(s, blobs, title=f"log {len(blobs)}").show()
    return len(blobs)


def count_droplets_doh(img):
    stats = ImageStats(img)
    s = stretch_composite_histogram(img, stats)
    blobs = doh(s)
    if DEBUG:
        make_circles_fig(s, blobs).show()
    return len(blobs)


def main():
    # data_dir = Path("RawData")
    # image_fn = Path("R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.jpeg")
    # image_fp = data_dir / image_fn

    for image_fn in glob.glob("processed_images/*.TIF"):
        img_org = imread(image_fn)
        img_gray = rgb2gray(img_org)

        start = time.time()
        print("dog", count_droplets_dog(img_gray))
        print(time.time() - start)

        # start = time.time()
        # print("log", count_droplets_log(img_gray))
        # print(time.time() - start)

        # start = time.time()
        # print("doh", count_droplets_doh(img_gray))
        # print(time.time() - start)


if __name__ == "__main__":
    main()
