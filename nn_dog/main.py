import os
from pathlib import Path

import torch

from nn_dog import DEVICE, PIN_MEMORY
from nn_dog.data import PLIF, SimulPLIF

# noinspection PyUnresolvedReferences
from nn_dog.model import DifferenceOfGaussiansStandardConv

# noinspection PyUnresolvedReferences
from sk_image.blob import make_circles_fig


def test_simul():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1000000)
    with torch.no_grad():
        model = DifferenceOfGaussiansStandardConv(
            min_sigma=1,
            max_sigma=10,
            overlap=0.9,
            # threshold=0.012,
            threshold=0.112,
            prune=False,
            sigma_bins=20,
        ).to(DEVICE, non_blocking=PIN_MEMORY)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        img_tensor = screenshot[0]
        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
        mask, local_maxima = model(img_tensor.unsqueeze(0).unsqueeze(0))
        blobs = model.make_blobs(mask, local_maxima)
        make_circles_fig(screenshot[0], blobs).show()


def main(data_dir_path):
    plif_dataset = PLIF(plif_dir=Path(data_dir_path))
    plif_dataloader = torch.utils.data.DataLoader(
        plif_dataset, batch_size=1, pin_memory=PIN_MEMORY, num_workers=1
    )

    img_height, img_width = plif_dataset[0].squeeze(0).numpy().shape
    with torch.no_grad():
        model = DifferenceOfGaussiansStandardConv(
            min_sigma=1,
            max_sigma=10,
            overlap=0.9,
            threshold=0.012,
            prune=False,
            sigma_bins=20,
        ).to(DEVICE, non_blocking=PIN_MEMORY)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        for i, img_tensor in enumerate(plif_dataloader):
            img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
            mask, local_maxima = model(img_tensor.unsqueeze(0))
            blobs = model.make_blobs(mask, local_maxima)
            make_circles_fig(plif_dataset[0], blobs).show()
            break


if __name__ == "__main__":
    # torch_dog_img_test()
    # set_start_method("spawn")
    main("/home/max/dev_projects/MERF_FSP/data")
    # dog_train()
