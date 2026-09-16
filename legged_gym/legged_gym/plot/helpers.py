import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot_colored_trajectory(
    ax,
    x,
    y,
    cmap="viridis",
    norm=None,
    linewidth=1.5,
):
    """
    Plot a 2-D trajectory colored according to time.

    x, y: shape [T]
    """

    points = np.column_stack([x, y]).reshape(-1, 1, 2)

    segments = np.concatenate(
        [points[:-1], points[1:]],
        axis=1
    )

    if norm is None:
        norm = Normalize(0, len(x) - 1)

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
    )

    lc.set_array(np.arange(len(x) - 1))

    ax.add_collection(lc)

    return lc