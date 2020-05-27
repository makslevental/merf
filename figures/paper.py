import csv
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib.lines import Line2D
from scipy import spatial
from skimage import io

from sk_image.blob import make_circles_fig


def gpu_gpu_copy():
    n_lines = 30
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    gpu_standard_df = pd.read_csv("gpu_fftconv_with_img_copy_run_times.csv")
    gpu_standard_df = pd.DataFrame(
        gpu_standard_df.groupby(["n_bin", "max_sigma"]).mean().to_records()
    )
    gpu_df = pd.read_csv("gpu_fftconv_run_times.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    for i in range(2, 31):
        gpu_st = gpu_standard_df[gpu_standard_df["max_sigma"] == i]["time"].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        if len(gpu_st) == 0 or len(gpu) == 0:
            continue

        ax.plot(np.arange(3, 51), gpu_st[1:], "--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3, 51), gpu[1:], c=cmap.to_rgba(i + 1))

    # ax.set_yscale('log')
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    fig.legend(
        [cpuline, gpuline],
        ["with copy", "no copy"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    fig.show()
    # tikzplotlib.save("test.tex", figure=fig)


def gpu_gpu():
    n_lines = 30
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    gpu_standard_df = pd.read_csv("gpu_standardconv_run_times.csv")
    gpu_df = pd.read_csv("gpu_fftconv_run_times.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    for i in range(2, 31):
        gpu_st = gpu_standard_df[gpu_standard_df["max_sigma"] == i]["time"].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        if len(gpu_st) == 0 or len(gpu) == 0:
            continue

        ax.plot(np.arange(3, 51), gpu_st[1:], "--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3, 51), gpu[1:], c=cmap.to_rgba(i + 1))

    ax.set_yscale("log")
    ax.set_title("GPU-FFTConv vs. GPU-StandardConv performance")
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    fig.legend(
        [cpuline, gpuline],
        ["StandardConv", "FFTConv"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    fig.show()
    # tikzplotlib.save("test.tex", figure=fig)


def cpu_gpu():
    n_lines = 29
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    cpu_df = pd.read_csv("cpu_run_times.csv")
    cpu_df = pd.DataFrame(cpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    gpu_df = pd.read_csv("gpu_fftconv_run_times.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    gpu_copy_df = pd.read_csv("gpu_fftconv_with_img_copy_run_times.csv")
    gpu_copy_df = pd.DataFrame(
        gpu_copy_df.groupby(["n_bin", "max_sigma"]).mean().to_records()
    )

    for i in range(2, 31):
        # cpu = cpu_df[cpu_df['max_sigma'] == i]['time'].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        gpu_copy = gpu_copy_df[gpu_copy_df["max_sigma"] == i]["time"].values
        # if len(cpu) == 0 or len(gpu) == 0: continue

        # ax.plot(np.arange(2,51), cpu,"--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(2, 51), gpu * 1000, c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(2, 51), gpu_copy * 1000, ":", c=cmap.to_rgba(i + 1))

    # ax.set_yscale('log')
    ax.set_title("GPU vs. CPU performance")
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    gpu_copy_line = Line2D([0], [0], color="blue", linewidth=2, linestyle=":")
    fig.legend(
        [cpuline, gpuline, gpu_copy_line],
        ["cpu", "gpu", "gpu_copy"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    fig.show()
    # tikzplotlib.save("test.tex", figure=fig)


def cpu_gpu_standard():
    n_lines = 29
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    cpu_df = pd.read_csv("cpu_run_times.csv")
    cpu_df = pd.DataFrame(cpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    gpu_df = pd.read_csv("gpu_standard_run_times_with_img_copy.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())

    for i in range(3, 31):
        cpu = cpu_df[cpu_df["max_sigma"] == i]["time"].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        if len(cpu) == 0 or len(gpu) == 0:
            continue

        ax.plot(np.arange(2, 51), cpu, "--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(2, 51), gpu, c=cmap.to_rgba(i + 1))

    # ax.set_yscale('log')
    ax.set_title("GPU vs. CPU performance")
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    gpu_copy_line = Line2D([0], [0], color="blue", linewidth=2, linestyle=":")
    fig.legend(
        [cpuline, gpuline, gpu_copy_line],
        ["cpu", "gpu"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    # fig.show()
    tikzplotlib.save("test.tex", figure=fig)


# https://scipython.com/book/chapter-8-scipy/problems/p84/overlapping-circles/
def intersection_area(d, R, r):
    if d <= abs(R - r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r) ** 2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r ** 2, R ** 2, d ** 2
    alpha = np.arccos((d2 + r2 - R2) / (2 * d * r))
    beta = np.arccos((d2 + R2 - r2) / (2 * d * R))
    return (
        r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2 * alpha) + R2 * np.sin(2 * beta))
    )


def circle_iou(c1, c2):
    print(c1, c2)
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    inters = intersection_area(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), r1, r2)
    unio = (4 / 3) * np.pi * (r1 ** 3 + r2 ** 3)
    return inters / unio


# for each detection that has a confidence score > threshold:
#
#     among
#     the
#     ground - truths, choose
#     one
#     that
#     belongs
#     to
#     the
#     same
#     class and has the highest IoU with the detection
#
#
#     if no ground-truth can be chosen or IoU < threshold (e.g., 0.5):
#         the
#         detection is a
#         false
#         positive
#     else:
#         the
#         detection is a
#         true
#         positive


def accuracy():
    for i in range(1, 100 + 1):
        truth_fp = f"../simulation/test_data/truth{i}.csv"
        cpu_res = f"accuracy_results/cpu/screenshot{i}.png.res"
        gpu_res = f"accuracy_results/gpu/screenshot{i}.png.res"
        img_fp = f"../simulation/test_data/screenshot{i}.png"
        img = io.imread(img_fp, as_gray=True)

        truth_csv = pd.read_csv(truth_fp)
        cpu_res = np.loadtxt(cpu_res)
        for (x, y, r) in cpu_res:
            print(truth_csv.apply(lambda r: circle_iou((x,y,r), (r['x'], r['y'], r['r'])), axis=1))


        break


if __name__ == "__main__":
    # gpu_gpu()
    # gpu_gpu_copy()
    # cpu_gpu_copy()
    # cpu_gpu()
    # cpu_gpu_standard()
    accuracy()
