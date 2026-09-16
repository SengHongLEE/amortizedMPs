import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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

    "axes.linewidth": 1.2,

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

idx = "0"

contact_boolean = np.load(
    load_path + idx + "_contact_boolean.npy"
)[:, 0, :]   # shape: [T, 4]

# ============================================================
# Basic settings
# ============================================================
min_step = 3000
max_step = 4000
dt = 0.001

contact_boolean = contact_boolean[min_step:max_step]   # [T, 4]
T = contact_boolean.shape[0]

time = np.arange(T) * dt

leg_labels = ["FL", "FR", "RL", "RR"]

# 转置成 [4, T]，方便画图
contact_img = contact_boolean.T

# ============================================================
# Colormap: 0 = white, 1 = colored
# ============================================================
cmap = ListedColormap(["white", "tab:blue"])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(12, 3.6))

im = ax.imshow(
    contact_img,
    aspect="auto",
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
    extent=[time[0], time[-1], 3.5, -0.5],   # 让 FL 在最上面
)

# ============================================================
# Axes labels and ticks
# ============================================================
ax.set_xlabel("Time [s]", labelpad=8)
ax.set_ylabel("Leg", labelpad=8)

ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(leg_labels)

# 每条腿之间画分隔线，更清楚
for y in [0.5, 1.5, 2.5]:
    ax.axhline(y, color="gray", linewidth=0.8, alpha=0.5)

# 网格只保留 x 方向辅助观察
ax.grid(
    True,
    axis="x",
    linestyle="--",
    linewidth=0.6,
    alpha=0.25,
)

ax.tick_params(
    direction="out",
    length=5,
    width=1.1,
)

ax.margins(x=0)

plt.tight_layout()
plt.show()