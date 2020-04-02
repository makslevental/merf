import glob
import time
from pathlib import Path

from skimage.color import rgb2gray
from skimage.io import imread

from blob import dog, make_circles_fig
from preprocess import gamma, match


def count_droplets(img):
    m = match(img)
    blobs = dog(m)
    make_circles_fig(m, blobs).show()


def main():
    # data_dir = Path("RawData")
    # image_fn = Path("R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.jpeg")
    # image_fp = data_dir / image_fn

    for image_fn in glob.glob("RawData/*.TIF"):
        img_org = imread(image_fn)
        img_gray = rgb2gray(img_org)
        start = time.time()
        count_droplets(img_gray)
        print(time.time() - start)


if __name__ == "__main__":
    main()
