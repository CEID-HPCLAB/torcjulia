from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

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

workloads = {
    "Workload 1": workload1,
    "Workload 2": workload2,
    "Workload 3": workload3,
    "Workload 4": workload4
}

WORKLOAD_COLORS = {
    "Workload 1": "#1D4ED8",
    "Workload 2": "darkred",
    "Workload 3": "forestgreen",
    "Workload 4": "darkorange",
}

# When reproducing Figure 2 of the paper using results obtained on a platform
# other than the ARIS supercomputer, adjust the axis settings below accordingly.
# The values provided here correspond to those used to generate Figure 2 in the paper.

WORKLOAD_SETTINGS = {
    "Workload 1": {
        "ylim": (0, 773),
        "yticks": [30, 360, 540, 720, *range(0, 201, 100)]
    },
    "Workload 2": {
        "ylim": (0, 546),
        "yticks": [20, 80, 130, 200, 280, 396, 504]
    },
     "Workload 3": {
        "ylim": (0, 650),
        "yticks": [30, 468, *range(0, 301, 100), 612]
    },
    "Workload 4": {
        "ylim": (0, 2349),
        "yticks": [100, 300, 720, 1080, 1620, 2160]
    }
}

plt.rcParams['font.family'] = 'Ubuntu'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 14

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

fig, axes = plt.subplots(1, len(workloads), figsize = (14, 3.1), sharey = False, dpi = 600)

for ax, (name, times) in zip(axes, workloads.items()):
    color = WORKLOAD_COLORS[name]
    ranks = np.arange(1, len(workers) + 1)

    baseline = times[0]

    formatter = FuncFormatter(lambda x, _: '' if x == 0 else f'{x / 3600:.2g}h' if x >= 360 else f'{int(x)}s')
    
    ax.bar(ranks, times, color = color, edgecolor = 'black', linewidth = 1, width = 0.7, zorder = 3, alpha = 0.9)

    for i in range(1, len(times)):
        speedup = baseline / times[i]

        if name == "Workload 3":
            gap = 6.6
        elif name == "Workload 1":
            gap = 6.9
        elif name == "Workload 4":
            gap = 24.5
        else:
            gap = 6

        ax.text(ranks[i], times[i] + gap, f"{speedup:.2f}×", ha = "center",
                va = "bottom", fontsize = 12, fontweight = "bold", rotation = 90, color = "#D1D5DB")

        # ax.text(ranks[i], times[i] + gap, f"{speedup:.2f}×", ha = "center",
        #         va = "bottom", fontsize = 12, fontweight = "bold", rotation = 90, color = "#475569")

    ax.set_xlabel("MPI Ranks", fontsize = 16)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
    
    ax.set_xticks(ranks); ax.set_xticklabels(workers, fontsize = 13)
    ax.yaxis.set_major_formatter(formatter)

    settings = WORKLOAD_SETTINGS.get(name)

    if settings:
        ax.set_ylim(*settings["ylim"])
        ax.set_yticks(sorted(settings["yticks"]))

axes[0].set_ylabel("Wall-Clock Time", labelpad = 10, fontsize = 16)

legend_handles = [Patch(facecolor = color, edgecolor = 'black', alpha = 0.9, label = name) for name, color in WORKLOAD_COLORS.items()]

fig.legend(handles = legend_handles, loc = 'upper center', ncol = 4, frameon = True, 
           facecolor = '#1a1a1a', 
           edgecolor = 'white', 
           bbox_to_anchor = (0.5, 1.15), framealpha = 1.0, fontsize = 14.5, 
           columnspacing = 1.5, borderpad = 0.4)

# fig.legend(handles = legend_handles, loc = 'upper center', ncol = 4, frameon = True, 
#            facecolor = 'white',
#            edgecolor = '#1a1a1a', 
#            bbox_to_anchor = (0.5, 1.15), framealpha = 1.0, fontsize = 14.5, 
#            columnspacing = 1.5, borderpad = 0.4)

for ax in axes:
    ax.grid(True, axis = 'y', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('white')
        # spine.set_color('black')

plt.tight_layout()

# plt.savefig(FIGURES_PATH / "runtime.pdf", dpi = 600, bbox_inches = "tight", transparent = True)
plt.savefig(FIGURES_PATH / "runtime.png", dpi = 600, bbox_inches = "tight", transparent = True)