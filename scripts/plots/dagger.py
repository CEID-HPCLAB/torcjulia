import warnings

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from matplotlib.lines import Line2D
from matplotlib import font_manager

STATIC_REGULAR_NAMES = {"Ubuntu-R.ttf", "Ubuntu-Regular.ttf"}
STATIC_BOLD_NAMES = {"Ubuntu-B.ttf", "Ubuntu-Bold.ttf"}
FONT = "Ubuntu"

def configure_font():
    fm = font_manager.fontManager

    def static(names):
        return [f for f in fm.ttflist
                if f.name == "Ubuntu" and Path(f.fname).resolve().name in names]

    regular, bold = static(STATIC_REGULAR_NAMES), static(STATIC_BOLD_NAMES)

    if not regular:
        warnings.warn(
            "Static Ubuntu Regular font not found; falling back to DejaVu Serif. "
            "Figures may differ slightly from those in the paper. To reproduce them exactly, "
            "install the static Ubuntu font (on Ubuntu: 'sudo apt install fonts-ubuntu-classic'; "
            "otherwise: https://fonts.google.com/specimen/Ubuntu) and clear the Matplotlib cache "
            "('rm -rf ~/.cache/matplotlib')."
        )
        plt.rcParams["font.family"] = "DejaVu Serif"
        global FONT; FONT = "DejaVu Serif"
        return

    fm.ttflist = [f for f in fm.ttflist if f.name != "Ubuntu"] + regular[:1] + bold[:1]

    if hasattr(fm, "_findfont_cached"):
        fm._findfont_cached.cache_clear()

    plt.rcParams["font.family"] = "Ubuntu"

configure_font()

RESULTS_PATH = Path(Path(__file__).resolve().parent / "../../experiments/results").resolve()
ARIS_PATH = RESULTS_PATH / "aris"

FIGURES_PATH = (Path(__file__).resolve().parent / "../../figures").resolve()
FIGURES_PATH.mkdir(parents = True, exist_ok = True)

def find_workload(filename, runtime = "torcjulia"):
    for directory in (RESULTS_PATH, ARIS_PATH):
        path = directory / runtime / filename

        if path.exists():
            return path

    raise FileNotFoundError(f"{filename} not found")

WORKLOAD1_TORCJULIA_PATH = find_workload("hooke-jeeves.txt")
WORKLOAD4_TORCJULIA_PATH = find_workload("mandelbrot-set.txt")
WORKLOAD1_DAGGER_PATH = find_workload("hooke-jeeves.txt", runtime = "dagger")
WORKLOAD4_DAGGER_PATH = find_workload("mandelbrot-set.txt", runtime = "dagger")

def load_workload(path):
    workers, times = [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("|")

            procs = int(parts[0].split(",")[0].split(":")[1].strip())
            t = float(parts[1].split(":")[1].strip().split()[0])

            workers.append(procs)
            times.append(t)

    return np.array(workers[1:]), np.array(times[1:])


workloads = {
    "workload1_torcjulia": load_workload(WORKLOAD1_TORCJULIA_PATH),
    "workload4_torcjulia": load_workload(WORKLOAD4_TORCJULIA_PATH),
    "workload1_dagger": load_workload(WORKLOAD1_DAGGER_PATH),
    "workload4_dagger": load_workload(WORKLOAD4_DAGGER_PATH)
}

workers = np.array(list(set.intersection(
    *(set(data[0]) for data in workloads.values())
)))

workers.sort()

for name, (ranks, times) in workloads.items():
    mask = np.isin(ranks, workers)
    workloads[name] = times[mask]

workload1_torcjulia = workloads["workload1_torcjulia"]
workload4_torcjulia = workloads["workload4_torcjulia"]
workload1_dagger = workloads["workload1_dagger"]
workload4_dagger = workloads["workload4_dagger"]

workloads = {
    "Workload 1": {
        "torcjulia": workload1_torcjulia,
        "Dagger": workload1_dagger
    },
    "Workload 4": {
        "torcjulia": workload4_torcjulia,
        "Dagger": workload4_dagger
    }
}

WORKLOAD_COLORS = {
    "Workload 1":  "#1D4ED8",
    "Workload 4": "darkorange"
}

# When reproducing Figure 3 of the paper using results obtained on a platform
# other than the ARIS supercomputer, adjust the axis settings below accordingly.
# The values provided here correspond to those used to generate Figure 3 in the paper.

WORKLOAD_SETTINGS = {
    "Workload 1": {
        "ylim": (0, 400),
        "yticks": [30, 360, *range(0, 201, 100)]
    },
    "Workload 4": {
        "ylim": (0, 1245),
        "yticks": [100, 300, 720, 1080]
    }
}

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 17

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

fig, axes = plt.subplots(1, len(workloads), figsize = (8.5, 3.4), sharey = False, dpi = 600)

bar_width = 0.32; bar_gap = 0.042

for ax, (name, times) in zip(axes, workloads.items()):
    color = WORKLOAD_COLORS[name]
    ranks = np.arange(1, len(workers) + 1) * 0.85

    formatter = FuncFormatter(
        lambda x, _: '' if x == 0 else f'{x / 3600:.2g}h' if x >= 360 else f'{int(x)}s'
    )

    offset = (bar_width + bar_gap) / 2
    
    ax.bar(ranks - offset, times["torcjulia"], color = color, alpha = 0.9, edgecolor = 'black', linewidth = 1, width = bar_width, zorder = 3)

    ax.bar(ranks + offset, times["Dagger"], color = color, alpha = 0.4, edgecolor = 'black', linewidth = 1, width = bar_width, zorder = 3, hatch = '///')

    gap_settings = {"Workload 1": {3: 3, 4: 0.15, "default": 2}, "default": {3: 9.99, 4: 0.451, "default": 6.66}}

    gaps = gap_settings.get(name, gap_settings["default"])

    offset_settings = {"Workload 1": {2: 0.01, 3: 0.01, 4: 0.05, "default": -0.02},
                       "default": {1: 0.01, 2: 0.01, 3: 0.01, 4: 0.05, "default": -0.02}}

    offsets = offset_settings.get(name, offset_settings["default"])

    for i, (x, dagger_time, torcjulia_time) in enumerate(zip(ranks, times["Dagger"], times["torcjulia"])):
        improvement = (dagger_time - torcjulia_time) / dagger_time * 100
        
        gap = gaps.get(i, gaps["default"])
        offset = offsets.get(i, offsets["default"])

        ax.text(x + offset, max(dagger_time, torcjulia_time) + gap, rf"$\downarrow\!\!${improvement:.1f}%",
                ha = "center", va = "bottom", fontsize = 15 if FONT == "Ubuntu" else 13.5, fontweight = "bold", color = "#D1D5DB")

        # ax.text(x + offset, max(dagger_time, torcjulia_time) + gap, rf"$\downarrow\!\!${improvement:.1f}%",
        #         ha = "center", va = "bottom", fontsize = 15 if FONT == "Ubuntu" else 13.5, fontweight = "bold", color = "#475569")
    
    ax.set_xlabel("Processes", fontsize = 20, labelpad = 3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
    
    ax.set_xticks(ranks); ax.set_xticklabels(workers, fontsize = 17 if FONT == "Ubuntu" else 15.5)
    ax.yaxis.set_major_formatter(formatter)

    settings = WORKLOAD_SETTINGS.get(name)

    if settings:
        ax.set_ylim(*settings["ylim"])
        ax.set_yticks(sorted(settings["yticks"]))

axes[0].set_ylabel("Wall-Clock Time", labelpad = 7, fontsize = 20)

runtime_handles = [Patch(facecolor = 'gray', edgecolor = 'black', label = 'torcjulia', alpha = 0.7),
                   Patch(facecolor='gray', edgecolor ='black', hatch ='///', label ='Dagger')]

workload_handles = [Patch(facecolor = color, edgecolor = 'black', label = name, alpha = 0.9) for name, color in WORKLOAD_COLORS.items()]

legend_handles = workload_handles + runtime_handles

fig.legend(handles = legend_handles, loc = 'upper center', ncol = 4, frameon = True,
           facecolor = '#1a1a1a', 
           edgecolor = 'white',
           bbox_to_anchor = (0.5, 1.153), framealpha = 1.0, fontsize = 17, labelspacing = 0.25,
           columnspacing = 1.1, borderpad = 0.4, handletextpad = 0.5)

# fig.legend(handles = legend_handles, loc = 'upper center', ncol = 4, frameon = True,
#            facecolor = 'white',
#            edgecolor = '#1a1a1a',
#            bbox_to_anchor = (0.5, 1.153), framealpha = 1.0, fontsize = 17, labelspacing = 0.25,
#            columnspacing = 1.1, borderpad = 0.4, handletextpad = 0.5)

for ax in axes:
    ax.grid(True, axis = 'y', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('white')
        # spine.set_color('black')

plt.tight_layout()

# plt.savefig(FIGURES_PATH / "dagger.pdf", dpi = 600, bbox_inches = "tight", transparent = True)
plt.savefig(FIGURES_PATH / "dagger.png", dpi = 600, bbox_inches = "tight", transparent = True)