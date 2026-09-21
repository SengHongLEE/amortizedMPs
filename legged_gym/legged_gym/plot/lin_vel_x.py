import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

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

    "legend.fontsize": 14,

    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,

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
idx = '1'
update_idx = np.load(
    load_path + idx + "_update_idx.npy"
)

base_lin_vel = np.load(
    load_path + idx + "_base_lin_vel.npy"
)

command = np.load(
    load_path + idx + "_command.npy"
)


# ============================================================
# Data processing
# ============================================================
min_step = 0
max_step = 10001

# ------------------------------------------------------------
# Velocity
# ------------------------------------------------------------
lin_vel_x = base_lin_vel[min_step:max_step, 0, 0]
cmd = command[min_step:max_step, 0, 0]


# ------------------------------------------------------------
# Effective update frequency
# ------------------------------------------------------------
# 假设 update_idx shape = [T, env, N]
#
# 你的定义：
# update_freq = sum(50 / update_idx) * 4 / 12
update_freq = (
    np.sum(
        50.0 / update_idx[int(min_step * 0.05):int(max_step * 0.05), 0, :],
        axis=-1
    )
    * 4.0 / 12.0
)

# update_freq[200:300] = update_freq[300:400]
# update_freq[250:300] = update_freq[50:100]
# 此时 update_freq shape = [T]


# ============================================================
# Time axis
# ============================================================
T = len(lin_vel_x)
T_P = len(update_freq)
dt = 0.001
time = np.arange(T) * dt
time_policy = np.arange(T_P) * 20 * dt

# ============================================================
# Moving average for actual velocity
# ============================================================
block_size = 20

# 只保留完整的 block
n_blocks = len(lin_vel_x) // block_size
n_valid = n_blocks * block_size

# 截断尾部不足 20 个的数据
lin_vel_x_trim = lin_vel_x[:n_valid]
time_trim = time[:n_valid]

# [n_blocks, 20]
lin_vel_x_blocks = lin_vel_x_trim.reshape(
    n_blocks,
    block_size
)

time_blocks = time_trim.reshape(
    n_blocks,
    block_size
)

# 每 20 个速度点取平均
lin_vel_x_avg = np.mean(
    lin_vel_x_blocks,
    axis=1
)

# 对应时间也取每个 block 的平均值
# 即把点放在这 20 个样本的时间中心
time_avg = np.mean(
    time_blocks,
    axis=1
)

# ============================================================
# Step 2: zero-phase low-pass filtering
# ============================================================

# Original dt = 0.001 s
# after averaging every 20 points:
dt_avg = dt * block_size

# Effective sampling frequency
fs_avg = 1.0 / dt_avg

# Low-pass cutoff frequency
cutoff = 1.5      # Hz
order = 3

b, a = butter(
    order,
    cutoff / (0.5 * fs_avg),
    btype="low"
)

lin_vel_x_smooth = filtfilt(
    b,
    a,
    lin_vel_x_avg
)

# ============================================================
# Create figure: 2 x 1
# ============================================================
fig, axs = plt.subplots(
    2,
    1,
    figsize=(10, 7.5),
    sharex=True,
)

fig.subplots_adjust(
    left=0.13,
    right=0.97,
    bottom=0.11,
    top=0.97,
    hspace=0.16,
)


# ============================================================
# (a) Velocity tracking
# ============================================================
ax = axs[0]

# Actual velocity
ax.plot(
    time_avg,
    lin_vel_x_smooth,
    linewidth=2.4,
    label=r"Actual $v_x$",
)

# Command velocity
ax.plot(
    time,
    cmd,
    color="black",
    linestyle="--",
    linewidth=2.2,
    label=r"Command $v_x^{cmd}$",
)

ax.set_ylabel(
    r"Velocity [m/s]",
    labelpad=10,
)

ax.legend(
    frameon=False,
    loc="lower right",
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.30,
)

ax.tick_params(
    direction="out",
    length=5,
    width=1.1,
)

ax.margins(x=0)


# ============================================================
# (b) Update frequency
# ============================================================
ax = axs[1]
ax.step(
    time_policy,
    update_freq,
    where="post",
    linewidth=2.2,
    label=r"Update frequency",
)

ax.set_xlabel(
    "Time [s]",
    labelpad=8,
)

ax.set_ylabel(
    r"Update frequency [Hz]",
    labelpad=10,
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.30,
)

ax.tick_params(
    direction="out",
    length=5,
    width=1.1,
)

ax.margins(x=0)


# ============================================================
# Panel labels
# ============================================================
# axs[0].text(
#     -0.10,
#     1.02,
#     "(a)",
#     transform=axs[0].transAxes,
#     fontsize=16,
#     fontweight="bold",
#     va="bottom",
# )

# axs[1].text(
#     -0.10,
#     1.02,
#     "(b)",
#     transform=axs[1].transAxes,
#     fontsize=16,
#     fontweight="bold",
#     va="bottom",
# )


plt.show()

# idx = np.arange(len(update_freq)) 
# print(update_freq.flatten()[idx % 50 == 0])
# print(update_freq.flatten()[idx % 50 == 0].mean())
print(np.mean(update_freq))

print(np.sqrt(np.mean((np.square(lin_vel_x - cmd)))))