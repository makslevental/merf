import glob

import numpy as np
import torch
from PIL import Image
from skimage.filters import gaussian
from skimage.io import imread

from sk_image.enhance_contrast import stretch_composite_histogram
from nns.n2n.unet import UnetN2N


def pil_loader(path):
    img = Image.open(path)
    return img


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def fluore_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. Range stays the same.
    Only output one channel, if RGB, convert to grayscale as well.
    Currently data is 8 bit depth.

    Args:
        pic (PIL Image): Image to be converted to Tensor.
    Returns:
        Tensor: only one channel, Tensor type consistent with bit-depth.
    """
    # handle PIL Image
    if pic.mode == "I":
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == "I;16":
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == "F":
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == "1":
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        # all 8-bit: L, P, RGB, YCbCr, RGBA, CMYK
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == "YCbCr":
        nchannel = 3
    elif pic.mode == "I;16":
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    img = img.view(pic.size[1], pic.size[0], nchannel)

    if nchannel == 1:
        img = img.squeeze(-1).unsqueeze(0)
    elif pic.mode in ("RGB", "RGBA"):
        # RBG to grayscale:
        # https://en.wikipedia.org/wiki/Luma_%28video%29
        ori_dtype = img.dtype
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        img = (img[:, :, [0, 1, 2]].float() * rgb_weights).sum(-1).unsqueeze(0)
        img = img.to(ori_dtype)
    else:
        # other type not supported yet: YCbCr, CMYK
        raise TypeError("Unsupported image type {}".format(pic.mode))

    return img


model = UnetN2N(in_channels=1, out_channels=1)
model.load_state_dict(
    torch.load(
        f"/Users/maksim/dev_projects/merf/n2n/checkpoints/model_epoch400.pth",
        map_location="cpu",
    )
)
model.eval()


def denoise(img_orig):
    filtered_img = gaussian(img_orig, sigma=1, preserve_range=True)
    log_img = stretch_composite_histogram(filtered_img)
    img = crop_center(log_img, 2048, 2048)

    t_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    denoised = model(t_img.float())
    return denoised[0].detach().numpy()[0]


if __name__ == "__main__":
    for img_fp in glob.glob("../data/RawData/4-10-8/*.TIF"):
        img_orig = imread(img_fp)
        denoise(img_orig)
        # threshold(img_orig)
        break
