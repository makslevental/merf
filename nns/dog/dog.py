import os
from os.path import expanduser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

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


def dog_train_test():
    # this is a stupid hack because running remote and profiling messes with file paths
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../../simulation/screenshot.png"
    )
    screenshot = Trivial(img_path=image_pth, num_repeats=50)
    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )
    dog = DifferenceOfGaussians(
        min_sigma=1, max_sigma=40, prune=True, overlap=0.9, threshold=0.1
    ).to(DEVICE, non_blocking=PIN_MEMORY)

    for name, param in dog.named_parameters():
        if name == "threshold":
            print(name, param.data)
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(dog.parameters())
    for i, (img_tensor, truth_tensor) in enumerate(train_dataloader):
        optimizer.zero_grad()

        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
        truth_tensor = truth_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
        local_maxima, mask = dog(img_tensor, soft_mask=True)
        loss = criterion(mask.unsqueeze(0), truth_tensor)
        loss.backward()
        print(loss)
        optimizer.step()

    dog.eval()

    for img_tensor, truth_tensor in train_dataloader:
        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
        local_maxima, mask = dog(img_tensor)
        blobs = dog.make_blobs(mask, local_maxima)
        print(len(blobs))
        make_circles_fig(screenshot[0][0].squeeze(0).numpy(), blobs).show()
        break


def dog_train():
    # this is a stupid hack because running remote and profiling messes with file paths
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../../simulation/screenshot.png"
    )
    screenshot = Trivial(img_path=image_pth, num_repeats=1)
    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )
    dog = DifferenceOfGaussians(
        min_sigma=1, max_sigma=40, prune=True, overlap=0.9, threshold=0.1
    ).to(DEVICE, non_blocking=PIN_MEMORY)
    criterion = nn.NLLLoss(reduction="sum")
    for img_tensor, truth_tensor in train_dataloader:
        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
        truth_tensor = truth_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
        local_maxima, mask = dog(img_tensor)


def torch_dog(dataloader, **dog_kwargs):
    with torch.no_grad():
        dog = DifferenceOfGaussians(**dog_kwargs).to(DEVICE, non_blocking=PIN_MEMORY)
        for p in dog.parameters():
            p.requires_grad = False
        dog.eval()
        for img_tensor, truth_tensor in dataloader:
            img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
            image_max, mask = dog(img_tensor)
            blobs = dog.make_blobs(image_max, mask)
    return blobs


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

    blobs = torch_dog(
        train_dataloader,
        min_sigma=1,
        max_sigma=40,
        prune=True,
        overlap=0.9,
        threshold=0.1,
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

    for i, blobs in enumerate(torch_dog(plif_dataloader, prune=True)):
        print("blobs: ", len(blobs))
        make_circles_fig(plif_dataset[i].squeeze(0).numpy(), blobs).show()
        counts, bin_centers, _ = plt.hist([r for (_, _, r) in blobs], bins=256)
        plt.show()


if __name__ == "__main__":
    # torch_dog_img_test()
    # set_start_method("spawn")
    # main()
    # dog_train()
    dog_train_test()
