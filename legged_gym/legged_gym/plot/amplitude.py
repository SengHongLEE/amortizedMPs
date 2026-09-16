import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from helpers import plot_colored_trajectory


# ============================================================
# Global academic style
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,

    "axes.labelsize": 20,
    "axes.titlesize": 20,

    "xtick.labelsize": 15,
    "ytick.labelsize": 15,

    "legend.fontsize": 11,

    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,

    # Mathematical symbols
    "mathtext.fontset": "stix",

    # Better vector-PDF text handling
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Load data
# ============================================================
load_path = "/home/ubuntu/amortizedMPs/legged_gym/legged_gym/metrics/single_speed-20s/"
idx = '0'
maxstep = 15000
amp_np = np.load(load_path + idx + "_amplitude.npy")[:maxstep, 0, :]
mu_history = np.sqrt(np.load(load_path + idx + "_mu.npy"))[:maxstep, 0, :]

# ============================================================
# Basic settings
# ============================================================
T = amp_np.shape[0]
dt = 0.001
time = np.arange(T) * dt   # 如果你知道 dt，可以改成 time = np.arange(T) * dt

leg_names = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
leg_short = ["FL", "FR", "RL", "RR"]

# ============================================================
# Unified y-limits for comparison
# ============================================================
all_values = np.concatenate([amp_np.flatten(), mu_history.flatten()])
y_min = all_values.min()
y_max = all_values.max()
y_margin = 0.08 * max(y_max - y_min, 1e-6)
y_lim = (y_min - y_margin, y_max + y_margin)

# ============================================================
# Create figure: 4 x 1
# ============================================================
fig, axs = plt.subplots(
    4, 1,
    figsize=(10, 13),
    sharex=True,
    sharey=True
)

fig.subplots_adjust(
    left=0.12,
    right=0.97,
    bottom=0.08,
    top=0.96,
    hspace=0.30
)

# ============================================================
# Plot
# ============================================================
for leg_id in range(4):
    ax = axs[leg_id]

    # r(t)
    ax.plot(
        time,
        amp_np[:, leg_id],
        color="tab:blue",
        linewidth=2.2,
        label=r"Amplitude $r$"
    )

    # mu(t)
    ax.plot(
        time,
        mu_history[:, leg_id],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=r"Parameter $\mu$"
    )

    ax.set_ylim(*y_lim)

    ax.set_title(
        f"{leg_names[leg_id]} ({leg_short[leg_id]})",
        fontweight="bold",
        pad=8
    )

    ax.set_ylabel("Value", labelpad=10)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.3
    )

    ax.tick_params(
        direction="out",
        length=5,
        width=1.0
    )

    ax.legend(
        loc="upper right",
        frameon=False
    )

# Only the last subplot has x label
axs[-1].set_xlabel("Time [s]", labelpad=10)

plt.show()