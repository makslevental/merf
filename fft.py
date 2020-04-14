import glob
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy import fftpack
from skimage.feature import blob_dog
from skimage.filters import gaussian
from skimage.io import imread
from sklearn import decomposition

from enhance_contrast import stretch_composite_histogram, ImageStats

# o_binary = opening(binary)
# make_figure(o_binary).show()
# c_binary = closing(o_binary)
# make_figure(c_binary).show()
from n2n.main import denoise
from preprocess import make_figure


# show(s2)
# thresh = threshold_otsu(s2)
# print(thresh, len(s2[s2 >= thresh].ravel()) / len(s2.ravel()))
# binary = s2 >= thresh
# make_figure(binary).show()


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle="->", linewidth=2, shrinkA=0, shrinkB=0)
    ax.annotate("", v1, v0, arrowprops=arrowprops)


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def test_dog(img_orig, show_fig=True):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(
        nrows=5, ncols=2, figsize=(24, 48)
    )
    fig.tight_layout()
    ax1.axis("off")
    # ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")
    ax5.axis("off")
    ax6.axis("off")
    ax7.axis("off")
    ax8.axis("off")
    # ax9.axis("off")
    # ax10.axis("off")

    im_fft = np.fft.fftshift(fftpack.fft2(img_orig))
    power = np.abs(im_fft) / (np.sum(np.abs(im_fft) ** 2) / im_fft.size)
    ax1.imshow(power, norm=LogNorm())
    ax1.set_title("log power")

    threshed_power = power > 1e-5
    power_vals = np.hstack(
        [np.argwhere(threshed_power), power[threshed_power, np.newaxis]]
    )
    pca = decomposition.PCA(n_components=2)
    pca.fit_transform(power_vals)
    center = list(pca.mean_[:2])[::-1]
    for length, vector in zip(pca.explained_variance_, pca.components_):
        length = np.sqrt(length) * 10
        print(length)
        v = vector[:2][::-1] * length
        draw_vector(center, center + v, ax1)

    ax2.hist(np.log(power.ravel()), bins=256, density=True)
    ax2.set_title("power distribution")

    filtered_img = gaussian(img_orig, sigma=1)
    log_img = np.log(filtered_img)
    print(log_img.min(), log_img.max())
    ax3.imshow(log_img, cmap="gray")
    ax3.set_title("log scaled")

    blobs = blob_dog(log_img, max_sigma=10, min_sigma=5, threshold=1, overlap=0.5)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)
    ax4.imshow(log_img, cmap="gray")
    ax4.set_title(f"log blobs {len(blobs)}")
    for y, x, r in blobs:
        c = plt.Circle((x, y), r, color="red", linewidth=0.5, fill=False)
        ax4.add_patch(c)

    s2 = stretch_composite_histogram(filtered_img)
    ax5.imshow(s2, cmap="gray")
    ax5.set_title("constrast stretched")

    blobs = blob_dog(s2, max_sigma=10, min_sigma=5, threshold=0.02, overlap=0.5)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)
    ax6.imshow(s2, cmap="gray")
    ax6.set_title(f"contrast stretched blobs {len(blobs)}")
    for y, x, r in blobs:
        c = plt.Circle((x, y), r, color="red", linewidth=0.5, fill=False)
        ax6.add_patch(c)
    ax10.hist([r for _, _, r in blobs], bins=256, density=True)
    ax10.set_title("constrast stretched bubble distribution")

    denoised = denoise(s2)
    ax7.imshow(denoised, cmap="gray")
    ax7.set_title("n2n denoised")
    blobs = blob_dog(denoised, max_sigma=10, min_sigma=5, threshold=0.02, overlap=0.5)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)
    ax8.imshow(denoised, cmap="gray")
    ax8.set_title(f"denoised blobs {len(blobs)}")
    for y, x, r in blobs:
        c = plt.Circle((x, y), r, color="red", linewidth=0.5, fill=False)
        ax8.add_patch(c)

    print([r for _, _, r in blobs])
    ax9.hist([r for _, _, r in blobs], bins=256, density=True)
    ax9.set_title("denoised bubble distribution")

    if show_fig:
        fig.show()
    else:
        fig.savefig(f"processed_images/{img_fp}.png", dpi=96)
    plt.close(fig)


def threshold(img_orig):
    for threshold in np.arange(-1, 0, 0.1):
        filtered_img = gaussian(img_orig, sigma=1)
        log_img = np.log(filtered_img)
        diff = log_img - gaussian(log_img, sigma=4)
        stats = ImageStats(diff)
        print(stats.mode, threshold)
        log_img[diff <= threshold] = 0
        log_img[diff >= threshold] = 1
        make_figure(log_img).show()


if __name__ == "__main__":
    for img_fp in glob.glob("RawData/4-10-8/*.TIF"):
        img_orig = imread(img_fp)
        test_dog(img_orig)
        # threshold(img_orig)
        break
