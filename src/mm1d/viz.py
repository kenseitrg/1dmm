"""Plotting: synthetics, gathers, semblance panels with event overlays.

M2 scope: zero-offset trace panels -- impulse-response stems (gray, one stem
per resolvable event) under a variable-area synthetic wiggle -- and stacked
figures comparing the three wavefield variants.  Every figure-writing script
saves deterministic filenames under ``output/`` and shows interactively unless
``--no-show`` is given.
"""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from mm1d.reflectivity import ALL_VARIANTS, WavefieldVariant

__all__ = ["VARIANT_LABELS", "plot_trace_panel", "zero_offset_stack_figure"]

VARIANT_LABELS: Mapping[WavefieldVariant, str] = {
    "full": "full: primaries + surface + interbed",
    "no_surface": "no_surface: primaries + interbed",
    "primaries": "primaries only (single scattering)",
}
"""One-line description of each wavefield variant for panel titles."""

SPIKE_RELATIVE_THRESHOLD = 1.0e-3
"""Impulse-response stems are drawn for samples above this fraction of max|sample|."""


def plot_trace_panel(
    ax: Axes,
    t_s: NDArray[np.float64],
    impulse: NDArray[np.float64],
    synthetic: NDArray[np.float64],
    *,
    title: str = "",
) -> None:
    """Plot one zero-offset trace panel: impulse stems + variable-area synthetic.

    Parameters
    ----------
    ax : Axes
        Target axes.
    t_s : ndarray
        Sample times, s.
    impulse : ndarray
        Impulse response trace (drawn as gray stems for every sample above
        :data:`SPIKE_RELATIVE_THRESHOLD` of the trace maximum, so events are
        countable by eye).
    synthetic : ndarray
        Wavelet-convolved trace (black variable-area wiggle).
    title : str, optional
        Panel title.
    """
    peak = float(np.max(np.abs(impulse)))
    events = np.abs(impulse) > SPIKE_RELATIVE_THRESHOLD * peak
    ax.vlines(t_s[events], 0.0, impulse[events], color="0.55", lw=1.1)
    ax.plot(t_s, synthetic, color="k", lw=0.8)
    ax.fill_between(t_s, synthetic, 0.0, where=synthetic > 0.0, color="k")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_ylim(-1.15 * peak, 1.15 * peak)
    ax.set_ylabel("amplitude")
    if title:
        ax.set_title(title, fontsize=9)


def zero_offset_stack_figure(
    t_s: NDArray[np.float64],
    panels: Mapping[WavefieldVariant, tuple[NDArray[np.float64], NDArray[np.float64]]],
    *,
    suptitle: str,
) -> Figure:
    """Stacked comparison figure of the wavefield variants (canonical order).

    Parameters
    ----------
    t_s : ndarray
        Sample times, s (shared horizontal axis).
    panels : mapping
        Variant name -> ``(impulse, synthetic)``; missing variants of
        :data:`~mm1d.reflectivity.ALL_VARIANTS` are skipped.
    suptitle : str
        Figure title.

    Returns
    -------
    Figure
        Unsaved figure with one row per variant (``full`` on top).
    """
    present = [v for v in ALL_VARIANTS if v in panels]
    fig = plt.figure(figsize=(11.0, 1.15 + 1.85 * len(present)))
    axes = fig.subplots(len(present), 1, sharex=True, squeeze=False)[:, 0]
    for ax, variant in zip(axes, present, strict=True):
        impulse, synthetic = panels[variant]
        plot_trace_panel(ax, t_s, impulse, synthetic, title=VARIANT_LABELS[variant])
    axes[-1].set_xlabel("two-way time s")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig
