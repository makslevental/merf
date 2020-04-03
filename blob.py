from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import numpy as np
from skimage import color, morphology
from skimage.draw import circle_perimeter
from skimage.feature import canny, blob_dog
from skimage.filters import sobel
from skimage.io import imread
from skimage.transform import hough_circle, hough_circle_peaks

# Load picture and detect edges
from preprocess import gamma


def circle(image):
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

    # Detect two radii
    hough_radii = np.arange(20, 35, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=3
    )

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        image[circy, circx] = (220, 20, 20)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()


def segmentation(image):
    elevation_map = sobel(image)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(elevation_map, cmap=plt.cm.gray)
    ax.set_title("elevation map")
    ax.axis("off")
    plt.show()
    markers = np.zeros_like(image)
    markers[image < 30] = 1
    markers[image > 150] = 2

    fig, ax = plt.subplots(figsize=(4 * 4, 3 * 4))
    ax.imshow(markers, cmap=plt.cm.nipy_spectral)
    ax.set_title("markers")
    ax.axis("off")
    plt.show()
    segmentation = morphology.watershed(elevation_map, markers)

    fig, ax = plt.subplots(figsize=(4 * 4, 3 * 4))
    ax.imshow(segmentation, cmap=plt.cm.gray)
    ax.set_title("segmentation")
    ax.axis("off")
    plt.show()

    return segmentation


def dog(image):
    blobs_dog = blob_dog(image, max_sigma=10, min_sigma=5, threshold=0.02, overlap=.9)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    return blobs_dog


def make_circles_fig(image, blobs, dpi=96):
    px, py = image.shape
    fig = plt.figure(figsize=(py / numpy.float(dpi), px / numpy.float(dpi)))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], yticks=[], xticks=[], frame_on=False)
    ax.imshow(image, cmap="gray")
    for y, x, r in blobs:
        c = plt.Circle((x, y), r, color="red", linewidth=.5, fill=False)
        ax.add_patch(c)
    return fig


def main():
    data_pth = Path("RawData/")
    image_fn = Path("R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.TIF")
    image_pth = data_pth / image_fn

    img_org = imread(image_pth)

    g = gamma(img_org)
    blobs = dog(g)
    make_circles_fig(g, blobs).show()


if __name__ == "__main__":
    main()
