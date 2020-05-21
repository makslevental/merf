import csv
import os
import time
from pathlib import Path

import torch

from nn_dog import PIN_MEMORY, DEVICE
from nn_dog.data import SimulPLIF
from nn_dog.model import DifferenceOfGaussiansFFT, DifferenceOfGaussiansStandardConv
from sk_image.blob import cpu_blob_dog

min_bin = 2
max_bin = 50
min_sigma = 1
mx_sigma = 30


def cpu_run_times():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)
    with open("cpu_run_times.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])
        for max_sigma in range(min_sigma, mx_sigma + 1):
            for n_bin in range(min_bin, max_bin + 1):
                START = time.monotonic()
                cpu_blob_dog(
                    screenshot[0],
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    overlap=0.9,
                    threshold=0.012,
                    sigma_bins=n_bin,
                    prune=False,
                )
                res = [n_bin, max_sigma, time.monotonic() - START]
                writer.writerow(res)
                print(res)
            #     break
            # break


def gpu_fft_run_times():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)
    img_height, img_width = screenshot[0].squeeze(0).numpy().shape

    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )

    with open("gpu_fftconv_run_times.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])

        for max_sigma in range(min_sigma + 1, mx_sigma + 1):
            for n_bin in range(min_bin, max_bin + 1):
                with torch.no_grad():
                    model = DifferenceOfGaussiansFFT(
                        img_height=img_height,
                        img_width=img_width,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        overlap=0.9,
                        threshold=0.012,
                        prune=False,
                        sigma_bins=n_bin,
                    ).to(DEVICE, non_blocking=PIN_MEMORY)
                    for p in model.parameters():
                        p.requires_grad = False
                    model.eval()
                    torch.cuda.synchronize()

                    img_tensor = next(iter(train_dataloader))
                    img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
                    torch.cuda.synchronize()

                    START = time.monotonic()

                    mask, local_maxima = model(img_tensor)
                    m, l = next(zip(mask, local_maxima))
                    blobs = model.make_blobs(m, l)
                    torch.cuda.synchronize()

                    END = time.monotonic()

                    res = [n_bin, max_sigma, END - START]
                    writer.writerow(res)
                    print(res)
            #     break
            # break


def gpu_fft_run_times_with_img_copy():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)
    img_height, img_width = screenshot[0].squeeze(0).numpy().shape

    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )

    with open("gpu_fftconv_with_img_copy_run_times.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])

        for max_sigma in range(min_sigma + 1, mx_sigma + 1):
            for n_bin in range(min_bin, max_bin + 1):
                with torch.no_grad():
                    model = DifferenceOfGaussiansFFT(
                        img_height=img_height,
                        img_width=img_width,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        overlap=0.9,
                        threshold=0.012,
                        prune=False,
                        sigma_bins=n_bin,
                    ).to(DEVICE, non_blocking=PIN_MEMORY)
                    for p in model.parameters():
                        p.requires_grad = False
                    model.eval()
                    torch.cuda.synchronize()

                    START = time.monotonic()

                    img_tensor = next(iter(train_dataloader))
                    img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
                    torch.cuda.synchronize()

                    mask, local_maxima = model(img_tensor)
                    m, l = next(zip(mask, local_maxima))
                    blobs = model.make_blobs(m, l)
                    torch.cuda.synchronize()

                    END = time.monotonic()

                    res = [n_bin, max_sigma, END - START]
                    writer.writerow(res)
                    print(res)
            #     break
            # break


def gpu_standard_run_times():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)
    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )

    with open("gpu_standardconv_run_times.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])

        for max_sigma in range(min_sigma + 1, mx_sigma + 1):
            for n_bin in range(min_bin, max_bin + 1):
                with torch.no_grad():
                    model = DifferenceOfGaussiansStandardConv(
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        overlap=0.9,
                        threshold=0.012,
                        prune=False,
                        sigma_bins=n_bin,
                    ).to(DEVICE, non_blocking=PIN_MEMORY)
                    for p in model.parameters():
                        p.requires_grad = False
                    model.eval()

                    img_tensor = next(iter(train_dataloader))
                    img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
                    torch.cuda.synchronize()

                    START = time.monotonic()

                    mask, local_maxima = model(img_tensor.unsqueeze(0))
                    blobs = model.make_blobs(mask, local_maxima)
                    torch.cuda.synchronize()

                    END = time.monotonic()

                    res = [n_bin, max_sigma, END - START]
                    writer.writerow(res)
                    print(res)
            #     break
            # break


def gpu_standard_run_times_with_img_copy():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)

    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )

    with open("gpu_standard_run_times_with_img_copy.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])

        for max_sigma in range(min_sigma + 1, mx_sigma + 1):
            for n_bin in range(min_bin, max_bin + 1):
                with torch.no_grad():
                    model = DifferenceOfGaussiansStandardConv(
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        overlap=0.9,
                        threshold=0.012,
                        prune=False,
                        sigma_bins=n_bin,
                    ).to(DEVICE, non_blocking=PIN_MEMORY)
                    for p in model.parameters():
                        p.requires_grad = False
                    model.eval()

                    START = time.monotonic()


                    img_tensor = next(iter(train_dataloader))
                    img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
                    torch.cuda.synchronize()

                    mask, local_maxima = model(img_tensor.unsqueeze(0))
                    blobs = model.make_blobs(mask, local_maxima)
                    torch.cuda.synchronize()

                    END = time.monotonic()

                    res = [n_bin, max_sigma, END - START]
                    writer.writerow(res)
                    print(res)
            #     break
            # break


if __name__ == "__main__":
    print("gpu_fft_run_times")
    gpu_fft_run_times()
    print("gpu_standard_run_times")
    gpu_standard_run_times()

    print("gpu_standard_run_times_with_img_copy")
    gpu_standard_run_times_with_img_copy()
    print("gpu_fft_run_times_with_img_copy")
    gpu_fft_run_times_with_img_copy()

    print("cpu_run_times")
    cpu_run_times()
