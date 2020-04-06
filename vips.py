import time

import numpy as np
import pyvips
from PIL import Image
from skimage import img_as_float
from skimage.feature import peak_local_max
from skimage.feature.blob import _prune_blobs

from enhance_contrast import ImageStats, get_min_and_max
# map vips formats to np dtypes
from preprocess import show

format_to_dtype = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    "complex": np.complex64,
    "dpcomplex": np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    "uint8": "uchar",
    "int8": "char",
    "uint16": "ushort",
    "int16": "short",
    "uint32": "uint",
    "int32": "int",
    "float32": "float",
    "float64": "double",
    "complex64": "complex",
    "complex128": "dpcomplex",
}


# numpy array to vips image
def numpy2vips(a):
    height, width = a.shape
    bands = 1
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(
        linear.data, width, height, bands, dtype_to_format[str(a.dtype)]
    )
    return vi


# vips image to numpy array
def vips2numpy(vi, dtype=None, bands=1):
    if dtype is None:
        dtype = vi.format
    return np.ndarray(
        buffer=vi.write_to_memory(),
        dtype=format_to_dtype[dtype],
        shape=[vi.height, vi.width],
    )


def stretch_by_hand(image):
    image = vips2numpy(image)
    stats = ImageStats(image)
    low, high = get_min_and_max(stats, saturation=0.3)
    image = np.clip(image, low, high)

    image = numpy2vips(image)
    image = (image - low) * (stats.max / (high - low))
    return image


def vips_img_as_numpy_float(image):
    return img_as_float(vips2numpy(image).astype(np.uint16))


def dog(image, max_sigma=10, min_sigma=5, threshold=0.02, overlap=0.9, sigma_ratio=1.6, exclude_border=False):
    ndim = 2

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [image.gaussblur(s) for s in sigma_list]
    gaussian_images = [vips_img_as_numpy_float(g) for g in gaussian_images]

    # computing difference between two successive Gaussian blurred images
    # multiplying with average standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                  * np.mean(sigma_list[i]) for i in range(k)]

    image_cube = np.stack(dog_images, axis=-1)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * (ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=exclude_border)
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]
    sigmas_of_peaks = np.expand_dims(sigmas_of_peaks, axis=1)

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]
    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)


def main():
    for _ in range(10):
        start = time.time()
        image = pyvips.Image.new_from_file("RawData/R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.TIF",
                                           access='random', memory=True)
        stretched_img = stretch_by_hand(image)
        dog(stretched_img)
        print(time.time() - start)


def test_gauss_blur():
    pillow_img = Image.open("RawData/R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.TIF")
    np_img = np.asarray(pillow_img)
    stretched_img = stretch_by_hand(np_img)
    show(stretched_img)
    for i in range(1, 10):
        img = stretched_img.gaussblur(i)
        img.write_to_file(f'processed_images/blur{i}.jpg')


if __name__ == "__main__":
    main()
    # test_gauss_blur()

# image = pyvips.Image.new_from_file(, access='sequential')
# image *= [1, 2, 1]
# mask = pyvips.Image.new_from_array([[-1, -1, -1],
#                                     [-1, 16, -1],
#                                     [-1, -1, -1]
#                                    ], scale=8)
# image = image.conv(mask, precision='integer')

# dtype = image.dtype.type
#
# imin, imax = intensity_range(image, in_range)
# omin, omax = intensity_range(image, out_range, clip_negative=(imin >= 0))
#
# image = np.clip(image, imin, imax)
#
# if imin != imax:
#     image = (image - imin) / float(imax - imin)
# return np.asarray(image * (omax - omin) + omin, dtype=dtype)

# low = image.percent(saturation/200)
# high = image.percent(saturation/200)
# image = (image - low) * (255 / (high - low))
# image.write_to_file("newfile.jpg")
#
# image.write_to_file('x.jpg')
