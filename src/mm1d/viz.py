"""Plotting: synthetics, gathers, semblance panels with event overlays.

M2 scope: zero-offset trace panels -- impulse-response stems (gray, one stem
per resolvable event) under a variable-area synthetic wiggle -- and stacked
figures comparing the three wavefield variants.

M3 scope: angle gathers -- variable-density panels (angle vs two-way time)
with optional analytic primary-intercept overlays and the post-critical mute
boundary.  Every angle-gather panel states the amplitude convention
(plane-wave pressures, no geometric spreading).  Every figure-writing script
saves deterministic filenames under ``output/`` and shows interactively unless
``--no-show`` is given.

M4 scope: offset gathers -- variable-density panels (offset vs two-way time)
with parametric theoretical-moveout overlays ``(x(p), T(p))`` from Snell ray
tracing.  Panels state the 2-D line-source plane-wave synthesis convention
(sqrt-omega wavelet, ``1/sqrt(2 pi |dx/dp|)`` focusing, no 3-D spreading).

M5 scope: semblance panels -- variable-density images with the zero-offset
time vertical (increasing downward, the seismic-section convention, so panels
compare side by side in columns) and the stacking velocity horizontal;
optional reference overlays (Dix RMS curve of the primaries,
pinned-multiple velocity lines) and a ``[0, 1]`` colorbar.
"""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from mm1d.reflectivity import ALL_VARIANTS, WavefieldVariant

__all__ = [
    "VARIANT_LABELS",
    "SemblanceOverlay",
    "angle_gather_stack_figure",
    "offset_gather_stack_figure",
    "plot_angle_gather",
    "plot_offset_gather",
    "plot_semblance",
    "plot_trace_panel",
    "semblance_stack_figure",
    "zero_offset_stack_figure",
]

VARIANT_LABELS: Mapping[WavefieldVariant, str] = {
    "full": "full: primaries + surface + interbed",
    "no_surface": "no_surface: primaries + interbed",
    "primaries": "primaries only (single scattering)",
}
"""One-line description of each wavefield variant for panel titles."""

SPIKE_RELATIVE_THRESHOLD = 1.0e-3
"""Impulse-response stems are drawn for samples above this fraction of max|sample|."""

SemblanceOverlay = Mapping[
    str,
    tuple[NDArray[np.float64], ...] | tuple[NDArray[np.float64], NDArray[np.float64], str],
]
"""Overlay curves of a semblance panel: label -> ``(t0, v)`` or ``(t0, v, fmt)``."""


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


def plot_angle_gather(
    ax: Axes,
    angles_deg: NDArray[np.float64],
    t_s: NDArray[np.float64],
    traces: NDArray[np.float64],
    *,
    title: str = "",
    moveout: Mapping[str, NDArray[np.float64]] | None = None,
    mute_angle_deg: float | None = None,
) -> None:
    """Plot one angle-gather panel as a variable-density image.

    Parameters
    ----------
    ax : Axes
        Target axes.
    angles_deg : ndarray
        Incidence-angle grid, degrees (assumed uniform for pixel edges).
    t_s : ndarray
        Sample times, s (uniform, starting at 0).
    traces : ndarray
        Gather traces, shape ``(n_angles, n_samples)``.
    title : str, optional
        Panel title.
    moveout : mapping, optional
        Label -> analytic intercept time per angle, s (``NaN`` where
        undefined); drawn as dashed curves (e.g. primary events from
        :func:`mm1d.gathers.primary_intercept_times`).
    mute_angle_deg : float, optional
        Angle beyond which traces are muted (post-critical resonance region,
        :func:`mm1d.gathers.critical_mute_slowness`); drawn as a dotted
        boundary line with the muted side shaded.
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    times = np.asarray(t_s, dtype=np.float64)
    d_angle = angles[1] - angles[0] if angles.size > 1 else 1.0
    dt = times[1] - times[0] if times.size > 1 else 1.0
    extent = (
        angles[0] - d_angle / 2.0,
        angles[-1] + d_angle / 2.0,
        times[-1] + dt,  # bottom edge (imshow with origin='upper')
        0.0,  # top edge
    )
    peak = float(np.max(np.abs(traces)))
    scale = peak if peak > 0.0 else 1.0
    ax.imshow(
        traces.T,
        extent=extent,
        origin="upper",
        aspect="auto",
        cmap="seismic",
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
    )
    if moveout:
        for label, tau in moveout.items():
            tau = np.asarray(tau, dtype=np.float64)
            ax.plot(angles, tau, "k--", lw=0.9)
            valid = np.flatnonzero(np.isfinite(tau))
            if valid.size:
                ax.text(
                    angles[valid[0]] + 0.15 * d_angle,
                    tau[valid[0]] + 0.012 * extent[2],
                    label,
                    color="k",
                    fontsize=8,
                    va="bottom",
                )
    if mute_angle_deg is not None:
        ax.axvline(mute_angle_deg, color="k", ls=":", lw=1.0)
        ax.axvspan(mute_angle_deg, extent[1], color="0.5", alpha=0.15)
        # vertically centered inside the muted band: matplotlib aligns the
        # rotated bounding box against the anchor, so va="center" keeps the
        # label clear of both the bottom angle axis and the panel top
        ax.text(
            mute_angle_deg + 0.15 * d_angle,
            0.5 * (extent[2] + extent[3]),
            "muted (post-critical)",
            color="0.3",
            fontsize=7,
            rotation=90,
            va="center",
            ha="left",
        )
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylabel("two-way time s")
    if title:
        ax.set_title(title, fontsize=9)


def angle_gather_stack_figure(
    angles_deg: NDArray[np.float64],
    t_s: NDArray[np.float64],
    panels: Mapping[WavefieldVariant, NDArray[np.float64]],
    *,
    suptitle: str,
    moveout: Mapping[str, NDArray[np.float64]] | None = None,
    mute_angle_deg: float | None = None,
) -> Figure:
    """Stacked comparison figure of angle gathers (canonical variant order).

    Parameters
    ----------
    angles_deg : ndarray
        Incidence-angle grid, degrees.
    t_s : ndarray
        Sample times, s (shared vertical axis).
    panels : mapping
        Variant name -> gather traces ``(n_angles, n_samples)``; missing
        variants of :data:`~mm1d.reflectivity.ALL_VARIANTS` are skipped.
    suptitle : str
        Figure title.
    moveout : mapping, optional
        Label -> analytic intercept time per angle, s; see
        :func:`plot_angle_gather`.
    mute_angle_deg : float, optional
        Mute boundary angle, degrees; see :func:`plot_angle_gather`.

    Returns
    -------
    Figure
        Unsaved figure with one row per variant (``full`` on top), shared
        angle/time axes.
    """
    present = [v for v in ALL_VARIANTS if v in panels]
    fig = plt.figure(figsize=(11.0, 1.4 + 2.6 * len(present)))
    axes = fig.subplots(len(present), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    for ax, variant in zip(axes, present, strict=True):
        plot_angle_gather(
            ax,
            angles_deg,
            t_s,
            panels[variant],
            title=VARIANT_LABELS[variant],
            moveout=moveout,
            mute_angle_deg=mute_angle_deg,
        )
    axes[-1].set_xlabel("incidence angle deg")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def plot_offset_gather(
    ax: Axes,
    offsets_m: NDArray[np.float64],
    t_s: NDArray[np.float64],
    traces: NDArray[np.float64],
    *,
    title: str = "",
    moveout: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]] | None = None,
) -> None:
    """Plot one offset-gather panel as a variable-density image.

    Parameters
    ----------
    ax : Axes
        Target axes.
    offsets_m : ndarray
        Offset grid, m (assumed uniform for pixel edges; first trace at
        ``x = 0`` is the zero-offset location).
    t_s : ndarray
        Sample times, s (uniform, starting at 0).
    traces : ndarray
        Gather traces, shape ``(n_offsets, n_samples)``.
    title : str, optional
        Panel title.
    moveout : mapping, optional
        Label -> ``(x_s_m, t_s)`` parametric moveout curve of one event (from
        :func:`mm1d.gathers.path_offset_time` / :func:`mm1d.gathers.primary_offset_times`;
        ``NaN`` entries are skipped); drawn as dashed curves.
    """
    offsets = np.asarray(offsets_m, dtype=np.float64)
    times = np.asarray(t_s, dtype=np.float64)
    d_offset = offsets[1] - offsets[0] if offsets.size > 1 else 1.0
    dt = times[1] - times[0] if times.size > 1 else 1.0
    extent = (
        offsets[0] - d_offset / 2.0,
        offsets[-1] + d_offset / 2.0,
        times[-1] + dt,  # bottom edge (imshow with origin='upper')
        0.0,  # top edge
    )
    peak = float(np.max(np.abs(traces)))
    scale = peak if peak > 0.0 else 1.0
    ax.imshow(
        traces.T,
        extent=extent,
        origin="upper",
        aspect="auto",
        cmap="seismic",
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
    )
    if moveout:
        for label, (x_curve, t_curve) in moveout.items():
            x_curve = np.asarray(x_curve, dtype=np.float64)
            t_curve = np.asarray(t_curve, dtype=np.float64)
            ax.plot(x_curve, t_curve, "k--", lw=0.9)
            valid = np.flatnonzero(np.isfinite(t_curve) & np.isfinite(x_curve))
            if valid.size:
                ax.text(
                    x_curve[valid[0]],
                    t_curve[valid[0]] + 0.012 * extent[2],
                    label,
                    color="k",
                    fontsize=8,
                    va="bottom",
                )
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylabel("two-way time s")
    if title:
        ax.set_title(title, fontsize=9)


def offset_gather_stack_figure(
    offsets_m: NDArray[np.float64],
    t_s: NDArray[np.float64],
    panels: Mapping[WavefieldVariant, NDArray[np.float64]],
    *,
    suptitle: str,
    moveout: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]] | None = None,
) -> Figure:
    """Stacked comparison figure of offset gathers (canonical variant order).

    Parameters
    ----------
    offsets_m : ndarray
        Offset grid, m (shared horizontal axis).
    t_s : ndarray
        Sample times, s (shared vertical axis).
    panels : mapping
        Variant name -> gather traces ``(n_offsets, n_samples)``; missing
        variants of :data:`~mm1d.reflectivity.ALL_VARIANTS` are skipped.
    suptitle : str
        Figure title.
    moveout : mapping, optional
        Label -> ``(x, t)`` moveout curves; see :func:`plot_offset_gather`.

    Returns
    -------
    Figure
        Unsaved figure with one row per variant (``full`` on top), shared
        offset/time axes.
    """
    present = [v for v in ALL_VARIANTS if v in panels]
    fig = plt.figure(figsize=(11.0, 1.4 + 2.6 * len(present)))
    axes = fig.subplots(len(present), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    for ax, variant in zip(axes, present, strict=True):
        plot_offset_gather(
            ax,
            offsets_m,
            t_s,
            panels[variant],
            title=VARIANT_LABELS[variant],
            moveout=moveout,
        )
    axes[-1].set_xlabel("offset m")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def plot_semblance(
    ax: Axes,
    v_grid_m_s: NDArray[np.float64],
    t0_grid_s: NDArray[np.float64],
    panel: NDArray[np.float64],
    *,
    title: str = "",
    overlay: SemblanceOverlay | None = None,
    colorbar: bool = True,
) -> AxesImage:
    """Plot one semblance panel: time vertical, stacking velocity horizontal.

    Zero-offset time increases downward (the seismic-section convention, as
    the gather panels of M3/M4), so several panels can be arranged in
    columns side by side (:func:`semblance_stack_figure`).

    Parameters
    ----------
    ax : Axes
        Target axes.
    v_grid_m_s : ndarray
        Stacking-velocity scan grid, m/s (assumed uniform for pixel edges;
        horizontal axis, increasing to the right).
    t0_grid_s : ndarray
        Zero-offset scan grid, s (assumed uniform for pixel edges; vertical
        axis, 0 at the top, increasing downward).
    panel : ndarray
        Semblance values, shape ``(n_v, n_t0)`` (from
        :func:`mm1d.semblance.semblance_panel`), range ``[0, 1]``.
    title : str, optional
        Panel title.
    overlay : mapping, optional
        Label -> ``(t0_s, v_m_s)`` reference curve (e.g. the Dix RMS curve of
        the primaries, or a pinned-multiple velocity line as a 2-point
        segment); drawn as white dashed curves. An optional third element
        sets the matplotlib format string (e.g. ``(t0, v, "w:")`` dotted for
        velocity pins, to distinguish them from curves).
    colorbar : bool, optional
        Add a labeled colorbar for the ``[0, 1]`` semblance scale to the right
        of the axes (default True; disable inside shared-axis stack figures).

    Returns
    -------
    AxesImage
        The image object (for figure-level colorbars in stack figures).
    """
    t0_grid = np.asarray(t0_grid_s, dtype=np.float64)
    v_grid = np.asarray(v_grid_m_s, dtype=np.float64)
    d_t0 = t0_grid[1] - t0_grid[0] if t0_grid.size > 1 else 1.0
    d_v = v_grid[1] - v_grid[0] if v_grid.size > 1 else 1.0
    extent = (
        v_grid[0] - d_v / 2.0,
        v_grid[-1] + d_v / 2.0,
        t0_grid[-1] + d_t0,  # bottom edge (imshow with origin='upper')
        0.0,  # top edge
    )
    image = ax.imshow(
        panel.T,
        extent=extent,
        origin="upper",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    if overlay:
        v_span = extent[1] - extent[0]
        t_span = extent[2] - extent[3]
        for label, curve in overlay.items():
            points = list(curve)
            t0_curve = np.asarray(points[0], dtype=np.float64)
            v_curve = np.asarray(points[1], dtype=np.float64)
            fmt = str(points[2]) if len(points) > 2 and isinstance(points[2], str) else "w--"
            ax.plot(v_curve, t0_curve, fmt, lw=1.0)
            valid = np.flatnonzero(np.isfinite(t0_curve) & np.isfinite(v_curve))
            if valid.size:
                ax.text(
                    v_curve[valid[0]] + 0.02 * v_span,
                    t0_curve[valid[0]] + 0.012 * t_span,
                    label,
                    color="w",
                    fontsize=8,
                    ha="left",
                    va="top",
                )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("stacking velocity m/s")
    ax.set_ylabel("zero-offset time s")
    if colorbar:
        ax.figure.colorbar(image, ax=ax, label="semblance")
    if title:
        ax.set_title(title, fontsize=9)
    return image


def semblance_stack_figure(
    v_grid_m_s: NDArray[np.float64],
    t0_grid_s: NDArray[np.float64],
    panels: Mapping[WavefieldVariant, NDArray[np.float64]],
    *,
    suptitle: str,
    overlay: SemblanceOverlay | None = None,
) -> Figure:
    """Column comparison figure of semblance panels (canonical variant order).

    Zero-offset time is the shared vertical axis (seismic convention), so the
    variants compare side by side; panels can also be stacked with gather
    figures of the same orientation.

    Parameters
    ----------
    v_grid_m_s : ndarray
        Stacking-velocity scan grid, m/s (shared horizontal axis).
    t0_grid_s : ndarray
        Zero-offset scan grid, s (shared vertical axis).
    panels : mapping
        Variant name -> semblance panel ``(n_v, n_t0)``; missing variants of
        :data:`~mm1d.reflectivity.ALL_VARIANTS` are skipped.
    suptitle : str
        Figure title.
    overlay : mapping, optional
        Label -> ``(t0, v[, fmt])`` reference curves; see :func:`plot_semblance`.

    Returns
    -------
    Figure
        Unsaved figure with one column per variant (``full`` leftmost), a
        shared time/velocity frame and one figure-level colorbar on the
        right (constrained layout reserves its space).
    """
    present = [v for v in ALL_VARIANTS if v in panels]
    # constrained layout: the figure-level colorbar's space is reserved
    # correctly (tight_layout re-arranges the parent panels and can drop the
    # colorbar on top of the last panel)
    fig = plt.figure(figsize=(1.9 + 3.1 * len(present), 6.0), layout="constrained")
    axes = fig.subplots(1, len(present), sharex=True, sharey=True, squeeze=False)[0, :]
    image = None
    for ax, variant in zip(axes, present, strict=True):
        image = plot_semblance(
            ax,
            v_grid_m_s,
            t0_grid_s,
            panels[variant],
            title=VARIANT_LABELS[variant],
            overlay=overlay,
            colorbar=False,
        )
    assert image is not None  # present is non-empty by ALL_VARIANTS construction
    fig.colorbar(image, ax=axes.tolist(), location="right", pad=0.02, label="semblance")
    fig.suptitle(suptitle)
    return fig


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
