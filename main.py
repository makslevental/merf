import time
from pathlib import Path

from skimage.color import rgb2gray
from skimage.io import imread

from blob import dog, make_circles_fig
from preprocess import gamma


def count_droplets(img):
    g = gamma(img)
    blobs = dog(g)
    make_circles_fig(g, blobs).show()


def main():
    data_dir = Path("RawData")
    image_fn = Path("R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.TIF")
    image_fp = data_dir / image_fn
    img_org = imread(image_fp)
    img_gray = rgb2gray(img_org)
    start = time.time()
    count_droplets(img_gray)
    print(time.time() - start)


if __name__ == "__main__":
    main()
