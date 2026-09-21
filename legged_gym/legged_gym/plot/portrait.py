import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# ============================================================
# Global academic style
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,

    "axes.labelsize": 18,
    "axes.titlesize": 19,

    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "legend.fontsize": 12,

    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,

    "mathtext.fontset": "stix",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Load data
# ============================================================
load_path = (
    "/home/ubuntu/amortizedMPs/legged_gym/"
    "legged_gym/metrics/EXP_MP/"
)
idx = '2'
max_step = 15000
amp_np = np.load(
    load_path + idx + "_amplitude.npy"
)[:max_step, 0, :]

phase_np = np.load(
    load_path + idx + "_phase.npy"
)[:max_step, 0, :]


# ============================================================
# Basic variables
# ============================================================
T = phase_np.shape[0]

# 根据你的实际记录间隔修改
dt = 0.001

time = np.arange(T) * dt


leg_names = [
    "Front Left (FL)",
    "Front Right (FR)",
    "Rear Left (RL)",
    "Rear Right (RR)",
]


# ============================================================
# Oscillator states
# ============================================================
osc_x = amp_np * np.cos(phase_np)
osc_y = amp_np * np.sin(phase_np)


# ============================================================
# Three representative stages
# ============================================================
# 建议选择代表性的局部窗口，而不是整段稳定阶段
stage_windows = [
    (2., 5.),   # Stage I
    (4.5, 7.5),   # Stage II
    (7., 10.),   # Stage III
]

stage_names = [
    "Before Transition",
    "Transition",
    "After Transition",
]


# ============================================================
# Unified oscillator limits
# ============================================================
osc_lim = max(
    np.max(np.abs(osc_x)),
    np.max(np.abs(osc_y)),
)

osc_lim *= 1.10


# ============================================================
# Helper:
# time-colored trajectory using absolute time
# ============================================================
def plot_phase_trajectory(
    ax,
    x,
    y,
    t,
    cmap,
    norm,
    linewidth=2.0,
):
    """
    Draw x-y trajectory with absolute time encoded by color.
    """

    points = np.column_stack(
        [x, y]
    ).reshape(-1, 1, 2)

    segments = np.concatenate(
        [
            points[:-1],
            points[1:]
        ],
        axis=1
    )

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
    )

    # color each segment according to absolute time
    lc.set_array(t[:-1])

    ax.add_collection(lc)

    return lc


# ============================================================
# Shared time normalization
# ============================================================
# 只覆盖三个选定窗口的总时间范围
time_norm = Normalize(
    vmin=stage_windows[0][0],
    vmax=stage_windows[-1][1],
)

cmap = "viridis"


# ============================================================
# Create 4 x 3 figure
# ============================================================
fig, axs = plt.subplots(
    4,
    3,
    figsize=(11.5, 13.5),
)

fig.subplots_adjust(
    left=0.12,
    right=0.89,
    bottom=0.075,
    top=0.93,
    wspace=0.28,
    hspace=0.30,
)


last_lc = None


# ============================================================
# Plot
# ============================================================
for leg_id in range(4):

    for stage_id, (t_start, t_end) in enumerate(stage_windows):

        ax = axs[leg_id, stage_id]

        # ----------------------------------------------------
        # Select current stage
        # ----------------------------------------------------
        mask = (
            (time >= t_start)
            & (time <= t_end)
        )

        stage_x = osc_x[mask, leg_id]
        stage_y = osc_y[mask, leg_id]
        stage_t = time[mask]

        # ----------------------------------------------------
        # Colored phase portrait
        # ----------------------------------------------------
        last_lc = plot_phase_trajectory(
            ax,
            stage_x,
            stage_y,
            stage_t,
            cmap=cmap,
            norm=time_norm,
            linewidth=2.2,
        )

        # ----------------------------------------------------
        # Start point
        # ----------------------------------------------------
        ax.scatter(
            stage_x[0],
            stage_y[0],
            s=45,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=1.3,
            zorder=5,
        )

        # ----------------------------------------------------
        # End point
        # ----------------------------------------------------
        ax.scatter(
            stage_x[-1],
            stage_y[-1],
            s=45,
            marker="s",
            facecolor="black",
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )

        # ----------------------------------------------------
        # Unified axes
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.6,
            alpha=0.25,
        )

        ax.tick_params(
            direction="out",
            length=4,
            width=1.0,
        )

        # ----------------------------------------------------
        # Column titles
        # ----------------------------------------------------
        if leg_id == 0:
            ax.set_title(
                stage_names[stage_id],
                fontweight="bold",
                pad=12,
            )

        # ----------------------------------------------------
        # Y label + leg name only on first column
        # ----------------------------------------------------
        if stage_id == 0:

            ax.set_ylabel(
                leg_names[leg_id]
                + "\n"
                + r"$r\sin\theta$",
                labelpad=12,
            )

        # ----------------------------------------------------
        # X label only on bottom row
        # ----------------------------------------------------
        if leg_id == 3:

            ax.set_xlabel(
                r"$r\cos\theta$",
                labelpad=7,
            )


# ============================================================
# Shared colorbar
# ============================================================
cbar_ax = fig.add_axes([
    0.92,    # left
    0.18,    # bottom
    0.018,   # width
    0.62,    # height
])

cbar = fig.colorbar(
    last_lc,
    cax=cbar_ax,
)

cbar.set_label(
    "Time [s]",
    fontsize=16,
    labelpad=10,
)

cbar.ax.tick_params(
    labelsize=13,
)


# ============================================================
# Figure title
# ============================================================
# fig.suptitle(
#     "Transition of Intrinsic Oscillator Dynamics",
#     fontsize=20,
#     fontweight="bold",
#     y=0.975,
# )


plt.show()