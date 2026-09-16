import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Global academic style
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,

    "axes.labelsize": 20,
    "axes.titlesize": 18,

    "xtick.labelsize": 15,
    "ytick.labelsize": 15,

    "legend.fontsize": 12,

    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,

    "mathtext.fontset": "stix",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# Load data
# ============================================================
load_path = "/home/ubuntu/amortizedMPs/legged_gym/legged_gym/metrics/EXP_MP/"
idx = '0'
max_step = 15000
phase_np = np.load(load_path + idx + "_phase.npy")[:max_step, 0, :]   # shape [T, 4]

# ============================================================
# Basic settings
# ============================================================
T = phase_np.shape[0]
dt = 0.001
time = np.arange(T) * dt   # 如果知道 dt，可改为 np.arange(T) * dt

leg_names = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
leg_short = ["FL", "FR", "RL", "RR"]

# ============================================================
# Helper functions
# ============================================================
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi
def wrap_to_2pi(x):
    return np.mod(x, 2 * np.pi)


# ============================================================
# Phase processing
# ============================================================
# FL absolute phase: unwrap for readability
phase_fl = np.unwrap(phase_np[:, 0])

# Relative phase differences w.r.t. FL
phase_diff_fr = wrap_to_2pi(phase_np[:, 1] - phase_np[:, 0])   # FR - FL
phase_diff_rl = wrap_to_2pi(phase_np[:, 2] - phase_np[:, 0])   # RL - FL
phase_diff_rr = wrap_to_pi(phase_np[:, 3] - phase_np[:, 0])   # RR - FL

phase_diff_all = [
    phase_diff_fr,
    phase_diff_rl,
    phase_diff_rr,
]

diff_titles = [
    r"Phase difference: FR relative to FL",
    r"Phase difference: RL relative to FL",
    r"Phase difference: RR relative to FL",
]

diff_labels = [
    r"$\theta_{\mathrm{FR}}-\theta_{\mathrm{FL}}$",
    r"$\theta_{\mathrm{RL}}-\theta_{\mathrm{FL}}$",
    r"$\theta_{\mathrm{RR}}-\theta_{\mathrm{FL}}$",
]

diff_colors = [
    "tab:orange",
    "tab:green",
    "tab:red",
]

# ============================================================
# Create figure: 4 x 1
# ============================================================
fig, axs = plt.subplots(
    4, 1,
    figsize=(10, 13),
    sharex=True
)

fig.subplots_adjust(
    left=0.14,
    right=0.97,
    bottom=0.08,
    top=0.97,
    hspace=0.32
)

# ============================================================
# Subplot 1: FL absolute phase
# ============================================================
ax = axs[0]

ax.plot(
    time,
    phase_fl,
    color="tab:blue",
    linewidth=2.2,
    label=r"Unwrapped phase $\theta_{\mathrm{FL}}$"
)

ax.set_title(
    "Front Left (FL) absolute phase",
    fontweight="bold",
    pad=8
)

ax.set_ylabel("Phase [rad]", labelpad=10)

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

# ============================================================
# Subplots 2-4: relative phase differences
# ============================================================
for i in range(3):
    ax = axs[i + 1]

    ax.plot(
        time,
        phase_diff_all[i],
        color=diff_colors[i],
        linewidth=2.2,
        label=diff_labels[i]
    )

    # Reference horizontal lines
    for y in [0, np.pi/2, np.pi, 1.5 * np.pi, 2 * np.pi]:
        ax.axhline(
            y=y,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.35
        )

    # ax.set_ylim(-0.5, 2 * np.pi + 0.5)

    ax.set_yticks([
        0,
        np.pi/2,
        np.pi,
        1.5 * np.pi,
        2. * np.pi,
    ])

    ax.set_yticklabels([
        r"$0$",
        r"$\pi/2$",
        r"$\pi$",
        r"$1.5 * \pi$",
        r"$2 * \pi$",
    ])

    ax.set_title(
        diff_titles[i],
        fontweight="bold",
        pad=8
    )

    ax.set_ylabel(r"$\Delta\theta$ [rad]", labelpad=10)

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

# ============================================================
# X label
# ============================================================
axs[-1].set_xlabel("Time [s]", labelpad=10)

plt.show()