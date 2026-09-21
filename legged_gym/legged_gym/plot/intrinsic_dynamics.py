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

    "axes.labelsize": 15,
    "axes.titlesize": 15,

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

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
load_path = "/home/ubuntu/amortizedMPs/legged_gym/legged_gym/metrics/EXP_MP/"
idx = '1'
max_step = 20000
amp_np = np.load(load_path + idx + "_amplitude.npy")[:max_step, 0, :]
phase_np = np.load(load_path + idx + "_phase.npy")[:max_step, 0, :]

# ============================================================
# Basic variables
# ============================================================
T = phase_np.shape[0]

dt = 0.001
total_time = T * dt

leg_names = [
    "Front Left",
    "Front Right",
    "Rear Left",
    "Rear Right",
]

leg_short_names = ["FL", "FR", "RL", "RR"]

# ============================================================
# Oscillator states
# ============================================================
osc_x = amp_np * np.cos(phase_np)
osc_y = amp_np * np.sin(phase_np)

# ============================================================
# Unified oscillator limits
# ============================================================
osc_lim = max(
    np.max(np.abs(osc_x)),
    np.max(np.abs(osc_y)),
)
osc_lim *= 1.10

# ============================================================
# Figure
# ============================================================
fig, axs = plt.subplots(
    2,
    2,
    figsize=(8.2, 7.2),
)
# Leave dedicated space for shared colorbar
fig.subplots_adjust(
    left=0.10,
    right=0.86,
    bottom=0.10,
    top=0.89,
    wspace=0.22,
    hspace=0.38,
)

# ============================================================
# Shared time normalization
# ============================================================
time_norm = Normalize(
    vmin=0,
    vmax=T - 1,
)

cmap = "viridis"
last_lc = None


# ============================================================
# Row 1
# Oscillator phase-plane trajectories
# ============================================================
for leg_id in [0,1]:

    ax = axs[0, leg_id]

    last_lc = plot_colored_trajectory(
        ax,
        osc_x[:, leg_id],
        osc_y[:, leg_id],
        cmap=cmap,
        norm=time_norm,
        linewidth=2.0,
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------
    ax.scatter(
        osc_x[0, leg_id],
        osc_y[0, leg_id],
        s=32,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------
    ax.scatter(
        osc_x[-1, leg_id],
        osc_y[-1, leg_id],
        s=32,
        marker="s",
        facecolor="black",
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )

    ax.set_xlim(
        -osc_lim,
        osc_lim,
    )

    ax.set_ylim(
        -osc_lim,
        osc_lim,
    )

    # 横纵坐标使用完全相同的刻度
    common_ticks = [-4, -2, 0, 2, 4]

    ax.set_xticks(common_ticks)
    ax.set_yticks(common_ticks)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    # Column title
    ax.set_title(
        leg_names[leg_id],
        pad=10,
        fontweight="bold",
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.30,
    )

    ax.tick_params(
        direction="out",
        length=4,
        width=1.0,
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------
    # ax.set_xlabel(
    #     r"$r\cos\theta$",
    #     labelpad=5,
    # )

    if leg_id == 0:
        ax.set_ylabel(
            r"$r\sin\theta$",
            labelpad=7,
        )

# ============================================================
# Row 2
# Oscillator phase-plane trajectories
# ============================================================
for leg_id in [2,3]:

    ax = axs[1, leg_id-2]

    last_lc = plot_colored_trajectory(
        ax,
        osc_x[:, leg_id],
        osc_y[:, leg_id],
        cmap=cmap,
        norm=time_norm,
        linewidth=2.0,
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------
    ax.scatter(
        osc_x[0, leg_id],
        osc_y[0, leg_id],
        s=32,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------
    ax.scatter(
        osc_x[-1, leg_id],
        osc_y[-1, leg_id],
        s=32,
        marker="s",
        facecolor="black",
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )

    ax.set_xlim(
        -osc_lim,
        osc_lim,
    )

    ax.set_ylim(
        -osc_lim,
        osc_lim,
    )

    # 横纵坐标使用完全相同的刻度
    common_ticks = [-4, -2, 0, 2, 4]

    ax.set_xticks(common_ticks)
    ax.set_yticks(common_ticks)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    # Column title
    ax.set_title(
        leg_names[leg_id],
        pad=10,
        fontweight="bold",
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.30,
    )

    ax.tick_params(
        direction="out",
        length=4,
        width=1.0,
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------
    ax.set_xlabel(
        r"$r\cos\theta$",
        labelpad=5,
    )

    if leg_id == 2:
        ax.set_ylabel(
            r"$r\sin\theta$",
            labelpad=7,
        )

# ============================================================
# Row labels
# ============================================================
# fig.text(
#     0.02,
#     0.7,
#     "(a)",
#     fontsize=15,
#     fontweight="bold",
#     va="center",
# )

# fig.text(
#     0.02,
#     0.285,
#     "(b)",
#     fontsize=15,
#     fontweight="bold",
#     va="center",
# )


# ============================================================
# Descriptive row titles
# ============================================================
# fig.text(
#     0.055,
#     0.665,
#     "Oscillator dynamics",
#     fontsize=13,
#     fontweight="bold",
#     rotation=90,
#     va="center",
#     ha="center",
# )

# fig.text(
#     0.055,
#     0.285,
#     "Foot trajectory",
#     fontsize=13,
#     fontweight="bold",
#     rotation=90,
#     va="center",
#     ha="center",
# )


# ============================================================
# Shared colorbar
# ============================================================
cbar_ax = fig.add_axes([
    0.885,       # left
    0.18,       # bottom
    0.022,      # width
    0.64,       # height
])

cbar = fig.colorbar(
    last_lc,
    cax=cbar_ax,
)

cbar.set_label(
    "Time [s]",
    fontsize=13,
    labelpad=10,
)

# ============================================================
# Convert sample index -> physical time [s]
# ============================================================
time_ticks_sec = np.arange(
    0,
    total_time + 1e-9,
    2.0,
)

# Corresponding positions in sample-index coordinates
time_ticks_idx = time_ticks_sec / dt

cbar.set_ticks(time_ticks_idx)
cbar.set_ticklabels(
    [f"{t:g}" for t in time_ticks_sec]
)

cbar.ax.tick_params(
    labelsize=11,
    length=4,
    width=1.0,
)


# ============================================================
# Main title
# ============================================================
# fig.suptitle(
#     "Oscillator Dynamics and Foot-End Trajectories",
#     fontsize=16,
#     fontweight="bold",
#     y=0.96,
# )


plt.show()