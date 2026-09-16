import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Global academic style
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,

    "axes.labelsize": 18,
    "axes.titlesize": 18,

    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "legend.fontsize": 12,

    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# Load data
# ============================================================
load_path = "/home/ubuntu/amortizedMPs/legged_gym/legged_gym/metrics/EXP_MP/"
idx = '1'
minstep = 1000
maxstep = 10001
ee_pos = np.load(load_path + idx + "_ee_pos.npy")   # shape: [T, env, leg, xyz]
base_pos = np.load(load_path + idx + "_base_position.npy")   # shape: [T, env, leg, xyz]

global_x = ee_pos[minstep:maxstep, 0, :, 0]   # [T, 4]
global_x[:, 0] = base_pos[minstep:maxstep, 0, 0]
global_x[:, 1] = base_pos[minstep:maxstep, 0, 0]
global_x[:, 2:] = global_x[:, :2]
global_z = ee_pos[minstep:maxstep, 0, :, 2]   # [T, 4]
# ============================================================
# Basic settings
# ============================================================
leg_names = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
leg_short = ["FL", "FR", "RL", "RR"]

# 给四条腿指定不同颜色
leg_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ============================================================
# Unified axis limits
# ============================================================
x_min = global_x.min()
x_max = global_x.max()

z_min = global_z.min()
z_max = global_z.max()

x_margin = 0.03 * max(x_max - x_min, 1e-6)
z_margin = 0.10 * max(z_max - z_min, 1e-6)

x_lim = (x_min - x_margin, x_max + x_margin)
z_lim = (z_min - z_margin, z_max + z_margin)

# ============================================================
# Create figure: 4 x 1
# ============================================================
fig, axs = plt.subplots(
    4, 1,
    figsize=(8, 14),
    sharex=True,
    sharey=True
)

fig.subplots_adjust(
    left=0.14,
    right=0.96,
    bottom=0.08,
    top=0.95,
    hspace=0.35
)

# ============================================================
# Plot each leg
# ============================================================
for leg_id in range(4):
    ax = axs[leg_id]
    # 轨迹曲线
    ax.plot(
        global_x[:, leg_id],
        global_z[:, leg_id],
        color=leg_colors[leg_id],
        linewidth=2.2,
        label=leg_short[leg_id]
    )

    # 起点
    ax.scatter(
        global_x[0, leg_id],
        global_z[0, leg_id],
        s=60,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
        label="Start" if leg_id == 0 else None
    )

    # 终点
    ax.scatter(
        global_x[-1, leg_id],
        global_z[-1, leg_id],
        s=60,
        marker="s",
        facecolor="black",
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
        label="End" if leg_id == 0 else None
    )

    # 可选：画地面线 z=0
    ax.axhline(
        y=0.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7
    )

    # 坐标范围统一
    ax.set_xlim(*x_lim)
    ax.set_ylim(*z_lim)

    # 标题
    ax.set_title(
        f"{leg_names[leg_id]} ({leg_short[leg_id]})",
        fontweight="bold",
        pad=10
    )

    # 纵轴标签
    ax.set_ylabel(r"Global $z$ [m]", labelpad=10)

    # 网格
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.3
    )

    ax.tick_params(
        direction="out",
        length=5,
        width=1.0
    )

    # 图例
    ax.legend(
        loc="best",
        frameon=False
    )

# 只有最后一个子图放横轴标签
axs[-1].set_xlabel(r"Global $x$ [m]", labelpad=10)

plt.show()