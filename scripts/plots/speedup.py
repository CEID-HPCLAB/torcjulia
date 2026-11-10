from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

RESULTS_PATH = Path(Path(__file__).resolve().parent / "../../experiments/results").resolve()
ARIS_PATH = RESULTS_PATH / "aris"

FIGURES_PATH = (Path(__file__).resolve().parent / "../../figures").resolve()
FIGURES_PATH.mkdir(parents = True, exist_ok = True)

def find_workload(filename):
    for directory in (RESULTS_PATH, ARIS_PATH):
        path = directory / "torcjulia" / filename

        if path.exists():
            return path

    raise FileNotFoundError(f"{filename} not found")

WORKLOAD1_PATH = find_workload("hooke-jeeves.txt")
WORKLOAD2_PATH = find_workload("monte-carlo.txt")
WORKLOAD3_PATH = find_workload("data-preprocessing.txt")
WORKLOAD4_PATH = find_workload("mandelbrot-set.txt")

def load_workload(path):
    workers, times = [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("|")

            mpi_ranks = int(parts[0].split(",")[0].split(":")[1].strip())
            threads = int(parts[0].split(",")[1].split(":")[1].strip())
            t = float(parts[1].split(":")[1].strip().split()[0])

            workers.append(mpi_ranks)
            times.append(t)

    return np.array(workers), np.array(times)

workers, workload1 = load_workload(WORKLOAD1_PATH)
_, workload2 = load_workload(WORKLOAD2_PATH)
_, workload3 = load_workload(WORKLOAD3_PATH)
_, workload4 = load_workload(WORKLOAD4_PATH)

datasets = [workload1, workload2, workload3, workload4]
labels = ["Workload 1", "Workload 2", "Workload 3", "Workload 4"]
colors = ["#1D4ED8", "darkred", "forestgreen", "darkorange"]

plt.rcParams['font.family'] = 'Ubuntu'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 12

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

def linear_scale(values, ticks):
    positions = np.arange(len(ticks))
    scaled = []
    for v in values:
        for i in range(len(ticks)-1):
            if ticks[i] <= v <= ticks[i+1]:
                scaled_val = positions[i] + (v - ticks[i]) / (ticks[i+1] - ticks[i])
                scaled.append(scaled_val)
                break
        else:
            scaled.append(positions[-1])
    return np.array(scaled)

speedups = [d[0]/d for d in datasets]

fig, ax = plt.subplots(figsize = (8.5, 5.1), dpi = 600)

for sp, c, l in zip(speedups, colors, labels):
    if l == "Workload 1":
        ax.plot(np.arange(len(workers)), linear_scale(sp, workers), marker = 'o', alpha = 0.9, color = c, label = l, linewidth = 2, zorder = 3)
    elif l == "Workload 3":
        ax.plot(np.arange(len(workers)), linear_scale(sp, workers), marker = 'o', alpha = 0.9, color = c, label = l, linewidth = 2, zorder = 2)
    else:
        ax.plot(np.arange(len(workers)), linear_scale(sp, workers), marker = 'o', alpha = 0.9, color = c, label = l, linewidth = 2, zorder = 1)

ax.plot(np.arange(len(workers)), np.arange(len(workers)), linestyle = '--', color = '#636363', label = 'Ideal', linewidth = 2)

ax.set_xlabel('MPI Ranks', fontsize = 20, labelpad = 10); ax.set_ylabel('Speedup', fontsize = 20)
ax.set_xticks(np.arange(len(workers))); ax.set_xticklabels(workers, fontsize = 20)
ax.set_yticks(np.arange(len(workers))); ax.set_yticklabels(workers, fontsize = 20)

dataset_handles = [Line2D([0], [0], color=colors[i], marker = 'o', lw = 2, alpha = 0.9) for i in range(len(colors))]

dataset_legend = ax.legend(handles = dataset_handles, labels = labels,
                           loc = 'upper left', frameon = True, 
                           facecolor = '#1a1a1a', edgecolor = 'white', 
                           ncols = 2, framealpha = 1.0, fontsize = 16.5, labelspacing = 0.09, 
                           columnspacing = 1.02, bbox_to_anchor = (0.003, 0.99))

# dataset_legend = ax.legend(handles = dataset_handles, labels = labels,
#                            loc = 'upper left', frameon = True, 
#                            facecolor = 'white', edgecolor = '#1a1a1a', 
#                            ncols = 2, framealpha = 1.0, fontsize = 16.5, labelspacing = 0.09, 
#                            columnspacing = 1.02, bbox_to_anchor = (0.003, 0.99))

ax.add_artist(dataset_legend)

ideal_handle = [Line2D([0], [0], color = '#636363', linestyle = '--', lw = 2, alpha = 0.9)]

ax.legend(handles = ideal_handle, labels = ['Ideal Speedup'], loc = 'upper left', frameon = True, 
          facecolor = '#1a1a1a', edgecolor = 'white',
          framealpha = 1.0, fontsize = 16.5, bbox_to_anchor = (0.005, 0.81))

# ax.legend(handles = ideal_handle, labels = ['Ideal Speedup'], loc = 'upper left', frameon = True, 
#           facecolor = 'white', edgecolor = '#1a1a1a', 
#           framealpha = 1.0, fontsize = 16.5, bbox_to_anchor = (0.005, 0.81))

ax.grid(True, axis = 'both', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('white')
    # spine.set_color('black')

ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()

# plt.savefig(FIGURES_PATH / "speedup.pdf", dpi = 600, bbox_inches = "tight", transparent = True)
plt.savefig(FIGURES_PATH / "speedup.png", dpi = 600, bbox_inches = "tight", transparent = True)