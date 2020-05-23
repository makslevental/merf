import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import tikzplotlib
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def gpu_gpu_copy():
    n_lines = 30
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    gpu_standard_df = pd.read_csv("gpu_fftconv_with_img_copy_run_times.csv")
    gpu_standard_df = pd.DataFrame(gpu_standard_df.groupby(['n_bin', 'max_sigma']).mean().to_records())
    gpu_df = pd.read_csv("gpu_fftconv_run_times.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(['n_bin', 'max_sigma']).mean().to_records())
    for i in range(2, 31):
        gpu_st = gpu_standard_df[gpu_standard_df['max_sigma'] == i]['time'].values
        gpu = gpu_df[gpu_df['max_sigma'] == i]['time'].values
        if len(gpu_st) == 0 or len(gpu) == 0: continue

        ax.plot(np.arange(3,51), gpu_st[1:],"--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3,51), gpu[1:], c=cmap.to_rgba(i + 1))

    # ax.set_yscale('log')
    ax.set_xticks([0,3,10,20,30,40,50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")


    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle='--')
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle='-')
    fig.legend([cpuline, gpuline], ["with copy", "no copy"], loc='upper left', bbox_to_anchor=(0.1, .9))
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
    gpu_df = pd.DataFrame(gpu_df.groupby(['n_bin', 'max_sigma']).mean().to_records())
    for i in range(2, 31):
        gpu_st = gpu_standard_df[gpu_standard_df['max_sigma'] == i]['time'].values
        gpu = gpu_df[gpu_df['max_sigma'] == i]['time'].values
        if len(gpu_st) == 0 or len(gpu) == 0: continue

        ax.plot(np.arange(3,51), gpu_st[1:],"--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3,51), gpu[1:], c=cmap.to_rgba(i + 1))

    ax.set_yscale('log')
    ax.set_title("GPU-FFTConv vs. GPU-StandardConv performance")
    ax.set_xticks([0,3,10,20,30,40,50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")


    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle='--')
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle='-')
    fig.legend([cpuline, gpuline], ["StandardConv", "FFTConv"], loc='upper left', bbox_to_anchor=(0.1, .9))
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
    cpu_df = pd.DataFrame(cpu_df.groupby(['n_bin', 'max_sigma']).mean().to_records())
    gpu_df = pd.read_csv("gpu_fftconv_run_times.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(['n_bin', 'max_sigma']).mean().to_records())
    gpu_copy_df = pd.read_csv("gpu_fftconv_with_img_copy_run_times.csv")
    gpu_copy_df = pd.DataFrame(gpu_copy_df.groupby(['n_bin', 'max_sigma']).mean().to_records())

    for i in range(2, 31):
        # cpu = cpu_df[cpu_df['max_sigma'] == i]['time'].values
        gpu = gpu_df[gpu_df['max_sigma'] == i]['time'].values
        gpu_copy = gpu_copy_df[gpu_copy_df['max_sigma'] == i]['time'].values
        # if len(cpu) == 0 or len(gpu) == 0: continue

        # ax.plot(np.arange(2,51), cpu,"--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(2,51), gpu*1000, c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(2,51), gpu_copy*1000, ":", c=cmap.to_rgba(i + 1))

    # ax.set_yscale('log')
    ax.set_title("GPU vs. CPU performance")
    ax.set_xticks([0,3,10,20,30,40,50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")


    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle='--')
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle='-')
    gpu_copy_line = Line2D([0], [0], color="blue", linewidth=2, linestyle=':')
    fig.legend([cpuline, gpuline, gpu_copy_line], ["cpu", "gpu", "gpu_copy"], loc='upper left', bbox_to_anchor=(0.1, .9))
    fig.show()
    # tikzplotlib.save("test.tex", figure=fig)

if __name__ == '__main__':
    # gpu_gpu()
    # gpu_gpu_copy()
    # cpu_gpu_copy()
    cpu_gpu()