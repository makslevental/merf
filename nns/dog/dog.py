import os
from os.path import expanduser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nns.dog.data import Trivial, PLIF
from nns.dog.model import DifferenceOfGaussians
from sk_image.blob import make_circles_fig
from sk_image.preprocess import make_figure

DATA_DIR = os.environ.get("FSP_DATA_DIR")
if DATA_DIR is None:
    raise Exception("need to specify env var FSP_DATA_DIR")
DATA_DIR = Path(expanduser(DATA_DIR))
NUM_GPUS = torch.cuda.device_count()
print(f"num gpus: {NUM_GPUS}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True


def torch_dog(dataloader, **dog_kwargs):
    with torch.no_grad():
        dog = DifferenceOfGaussians(**dog_kwargs).to(DEVICE, non_blocking=PIN_MEMORY)
        for p in dog.parameters():
            p.requires_grad = False
        dog.eval()
        for img_tensor in dataloader:
            img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
            mask, local_maxima = dog(img_tensor)
            blobs = dog.make_blobs(local_maxima, mask)
            yield blobs


def torch_dog_img_test():
    # this is a stupid hack because running remote and profiling messes with file paths
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../../simulation/screenshot.png"
    )
    screenshot = Trivial(img_path=image_pth, num_repeats=1)
    make_figure(screenshot[0][0].squeeze(0).numpy()).show()
    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )
    for t in np.linspace(0.1, 0.9, 10):
        blobs = torch_dog(
            train_dataloader,
            min_sigma=1,
            max_sigma=40,
            prune=True,
            overlap=0.9,
            threshold=t,
        )
        print(len(blobs))
        make_circles_fig(screenshot[0][0].squeeze(0).numpy(), blobs).show()
        plt.hist([r for (_, _, r) in blobs], bins=256)
        plt.show()


def main():
    plif_dataset = PLIF(plif_dir=DATA_DIR)
    plif_dataloader = torch.utils.data.DataLoader(
        plif_dataset, batch_size=1, pin_memory=PIN_MEMORY, num_workers=4
    )
    for t in np.linspace(0.01, 0.1, 10):
        for i, blobs in enumerate(
            torch_dog(
                plif_dataloader,
                min_sigma=1,
                max_sigma=20,
                overlap=0.9,
                threshold=t,
                prune=True,
            )
        ):
            print("blobs: ", len(blobs))
            make_circles_fig(plif_dataset[i].squeeze(0).numpy(), blobs).show()
            counts, bin_centers, _ = plt.hist([r for (_, _, r) in blobs], bins=256)
            plt.show()
            break


if __name__ == "__main__":
    # torch_dog_img_test()
    # set_start_method("spawn")
    main()
    # dog_train()
