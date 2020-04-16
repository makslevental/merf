import numpy as np
import pyvips
from skimage import img_as_float
from skimage.feature import peak_local_max
from skimage.feature.blob import _prune_blobs

from sk_image.enhance_contrast import ImageStats, get_min_and_max

# map vips formats to np dtypes
from sk_image.preprocess import make_figure

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

    image = (image - low) / (high - low)
    image = image * (stats.max - stats.min) + stats.min
    return image


def vips_img_as_numpy_float(image):
    return img_as_float(vips2numpy(image).astype(np.uint16))


DEBUG = True


def dog(
    image, threshold=0.001, overlap=0.8, min_sigma=1, max_sigma=10, sigma_ratio=1.6
):
    ndim = 2

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1)

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i) for i in range(k + 1)])
    print("sigmas: ", sigma_list)

    gaussian_images = [image.gaussblur(s) for s in sigma_list]
    gaussian_images = [vips_img_as_numpy_float(g) for g in gaussian_images]
    if DEBUG:
        for i, d in enumerate(gaussian_images):
            make_figure(d).savefig(
                f"/Users/maksim/dev_projects/merf/data/processed_images/debug/vips_gaussian_{i}.png"
            )

    # computing difference between two successive Gaussian blurred images
    # multiplying with average standard deviation provides scale invariance
    dog_images = [
        (gaussian_images[i] - gaussian_images[i + 1]) * sigma_list[i] for i in range(k)
    ]
    if DEBUG:
        for i, d in enumerate(dog_images):
            make_figure(d).savefig(
                f"/Users/maksim/dev_projects/merf/data/processed_images/debug/vips_diff_{i}.png"
            )

    image_cube = np.stack(dog_images, axis=-1)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        footprint=np.ones((3,) * (ndim + 1)),
        threshold_rel=0.0,
        exclude_border=False,
    )
    print("num maxes: ", len(local_maxima))
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
    image_pth = "/Users/maksim/dev_projects/merf/data/RawData/R-233_5-8-6_000110.T000.D000.P000.H000.PLIF1.TIF"
    image = pyvips.Image.new_from_file(image_pth, access="random", memory=True)
    image = image.gaussblur(1)
    stretched_img = stretch_by_hand(image)
    dog(stretched_img)
    # for g in glob.glob("../data/RawData/*.TIF"):
    #     start = time.time()
    #     image = pyvips.Image.new_from_file(g, access="random", memory=True, )
    #     stretched_img = stretch_by_hand(image)
    #     dog(stretched_img)
    #     # make_circles_fig(vips2numpy(stretched_img), ).show()
    #     print(time.time() - start)
    #     break

def test_ax():
    from scipy import ndimage, misc
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ascent = misc.ascent()
    result = ndimage.maximum_filter(ascent, size=20)
    ax1.imshow(ascent)
    ax2.imshow(result)
    plt.show()

if __name__ == "__main__":
    # main()
    test_ax()