import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import tikzplotlib
import matplotlib.patches as mpatches


n_lines = 29
c = np.arange(1, n_lines + 1)

norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])

fig, ax = plt.subplots(dpi=100)

cpu_df = pd.read_csv("cpu_run_times.csv")
gpu_df = pd.read_csv("gpu_fftconv_run_times.csv")

for i in range(2, 32):
    cpu = cpu_df[cpu_df['max_sigma'] == i]['time'].values[1:]
    gpu = gpu_df[gpu_df['max_sigma'] == i]['time'].values[1:]
    if len(cpu) == 0 or len(gpu) == 0: continue

    ax.plot(cpu,"--", c=cmap.to_rgba(i + 1))
    ax.plot(gpu, c=cmap.to_rgba(i + 1))

ax.set_yscale('logit')
ax.set_title("GPU vs. CPU performance")
ax.set_xticks([0,3,10,20,30,40,50])
ax.set_ylabel("time (s)")
ax.set_xlabel("filters")
fig.colorbar(cmap, ticks=c, label="max radius")
red_patch = mpatches.Patch(color='red', label='The red data', linestyle="--")
blue_patch = mpatches.Patch(color='blue', label='The blue data')
fig.legend(handles=[red_patch, blue_patch], loc='upper left', bbox_to_anchor=(0.1, .9))
fig.show()
# tikzplotlib.save("test.tex", figure=fig)
