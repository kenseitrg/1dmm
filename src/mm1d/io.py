"""CSV/TXT log loading, resampling, blocking.

Two input families are supported:

- **Sampled logs** -- rows are samples; ``vp``/``rho`` are treated as linearly
  interpolated between sample depths.  :func:`load_log` reads a single
  three-column file (``depth_m, vp_m_s, rho_kg_m3``); :func:`load_log_pair`
  merges two two-column files (``depth_m, vp_m_s`` and ``depth_m,
  rho_kg_m3``) on depth with a tolerance.  :func:`block_log_uniform_twt`
  converts a sampled log into a :class:`~mm1d.model.LayeredModel` with
  (near-)uniform layer two-way times, using exact slowness integration for
  linearly interpolated velocity (so layer velocities are harmonic means and
  layer TWTs are consistent by construction) and exact piecewise-linear
  averaging for density.
- **Blocked models** -- rows are layer tops with constant properties; the
  last row is the half-space.  :func:`load_blocked_model` reads the canonical
  models in ``data/`` (exact layer boundaries, no interpolation smearing).

All loaders validate: strictly increasing depth, finite values, positive
velocity/density; failures raise :class:`LogError` with the offending row.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mm1d.model import LayeredModel

__all__ = [
    "Log",
    "LogError",
    "block_log_uniform_twt",
    "load_blocked_model",
    "load_log",
    "load_log_pair",
]


class LogError(ValueError):
    """Raised for unreadable or invalid log files."""


# ---------------------------------------------------------------------------
# Parsing and validation helpers
# ---------------------------------------------------------------------------


def _parse_numeric_table(text: str, source: str) -> NDArray[np.float64]:
    """Parse a numeric table: one row per line, comma or whitespace separated.

    ``#`` comment lines and blank lines are skipped; a single unparseable line
    before the first numeric row is tolerated as a header.

    Raises
    ------
    LogError
        If unparseable data lines or inconsistent column counts are found, or
        if no numeric rows exist.
    """
    rows: list[list[float]] = []
    first_row_lineno = -1
    bad: list[int] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = [float(part) for part in line.replace(",", " ").split()]
        except ValueError:
            bad.append(lineno)
        else:
            if first_row_lineno < 0:
                first_row_lineno = lineno
            rows.append(row)
    if not rows:
        raise LogError(f"{source}: no numeric data rows found")
    if bad and not (len(bad) == 1 and first_row_lineno > 0 and bad[0] < first_row_lineno):
        details = ", ".join(str(no) for no in bad[:5])
        raise LogError(f"{source}: could not parse line(s) {details} as numbers")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise LogError(
            f"{source}: inconsistent column counts {sorted(widths)}; every data row "
            f"must have the same number of columns"
        )
    return np.asarray(rows, dtype=np.float64)


def _validate_depth(depth_m: NDArray[np.float64], source: str) -> None:
    """Require a finite, strictly increasing depth sample vector."""
    bad = np.flatnonzero(~np.isfinite(depth_m))
    if bad.size:
        raise LogError(f"{source}: non-finite depth value(s) at sample {bad[:5].tolist()}")
    if depth_m.size < 2:
        raise LogError(f"{source}: need at least 2 samples, got {depth_m.size}")
    bad = np.flatnonzero(np.diff(depth_m) <= 0.0)
    if bad.size:
        i = int(bad[0]) + 1
        raise LogError(
            f"{source}: depth must be strictly increasing; sample {i} "
            f"(depth {depth_m[i]}) follows depth {depth_m[i - 1]}"
        )


def _validate_property(values: NDArray[np.float64], label: str, source: str) -> None:
    """Require finite, strictly positive log values (velocity or density)."""
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        raise LogError(f"{source}: non-finite {label} value(s) at sample {bad[:5].tolist()}")
    bad = np.flatnonzero(values <= 0.0)
    if bad.size:
        raise LogError(f"{source}: {label} must be > 0; violation at sample {bad[:5].tolist()}")


# ---------------------------------------------------------------------------
# Sampled logs
# ---------------------------------------------------------------------------


class Log:
    """Depth-sampled log (SI units): depth ``m``, ``vp`` m/s, ``rho`` kg/m^3.

    A light validated record; velocities are meant to be linearly interpolated
    between samples by downstream blocking.
    """

    __slots__ = ("depth_m", "rho_kg_m3", "vp_m_s")

    def __init__(
        self,
        depth_m: NDArray[np.float64],
        vp_m_s: NDArray[np.float64],
        rho_kg_m3: NDArray[np.float64],
    ) -> None:
        """Store float64 copies of the sample vectors and validate them."""
        depth = np.array(depth_m, dtype=np.float64, copy=True)
        vp = np.array(vp_m_s, dtype=np.float64, copy=True)
        rho = np.array(rho_kg_m3, dtype=np.float64, copy=True)
        if depth.ndim != 1 or vp.ndim != 1 or rho.ndim != 1:
            raise LogError("Log inputs must be 1-D arrays")
        if not (depth.size == vp.size == rho.size):
            raise LogError(
                f"Log inputs must have equal length, got {depth.size}, {vp.size}, {rho.size}"
            )
        source = "Log"
        _validate_depth(depth, source)
        _validate_property(vp, "vp_m_s", source)
        _validate_property(rho, "rho_kg_m3", source)
        self.depth_m = depth
        self.vp_m_s = vp
        self.rho_kg_m3 = rho

    def __repr__(self) -> str:
        """Short one-line representation."""
        return (
            f"Log(n_samples={self.depth_m.size}, depth "
            f"{self.depth_m[0]:.1f}..{self.depth_m[-1]:.1f} m)"
        )


def load_log(path: str | Path) -> Log:
    """Load a single three-column sampled log: ``depth_m, vp_m_s, rho_kg_m3``.

    Comma or whitespace separated; an optional single header line and ``#``
    comments are skipped.

    Raises
    ------
    LogError
        On wrong column count, unparseable lines, non-monotonic depth,
        non-finite or non-positive values.
    """
    source = str(path)
    table = _parse_numeric_table(Path(path).read_text(encoding="utf-8"), source)
    if table.shape[1] != 3:
        raise LogError(
            f"{source}: expected 3 columns (depth_m, vp_m_s, rho_kg_m3), got {table.shape[1]}"
        )
    depth_m = table[:, 0]
    vp_m_s = table[:, 1]
    rho_kg_m3 = table[:, 2]
    _validate_depth(depth_m, source)
    _validate_property(vp_m_s, "vp_m_s", source)
    _validate_property(rho_kg_m3, "rho_kg_m3", source)
    return Log(depth_m, vp_m_s, rho_kg_m3)


def load_log_pair(vp_path: str | Path, rho_path: str | Path, depth_tolerance_m: float = 0.5) -> Log:
    """Load two two-column files and merge them on depth with a tolerance.

    Each file has columns ``depth_m, value`` -- velocity for ``vp_path``,
    density for ``rho_path``.  Every vp sample must have a rho sample within
    ``depth_tolerance_m``; the merged depth is the mean of the matched pair.

    Raises
    ------
    LogError
        On wrong column counts, invalid values, or unmatched depths beyond
        the tolerance.
    """
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0.0:
        raise LogError(f"depth_tolerance_m must be finite and >= 0, got {depth_tolerance_m}")
    source = f"{vp_path} + {rho_path}"
    vp_table = _parse_numeric_table(Path(vp_path).read_text(encoding="utf-8"), str(vp_path))
    rho_table = _parse_numeric_table(Path(rho_path).read_text(encoding="utf-8"), str(rho_path))
    if vp_table.shape[1] != 2:
        raise LogError(f"{vp_path}: expected 2 columns (depth_m, vp_m_s), got {vp_table.shape[1]}")
    if rho_table.shape[1] != 2:
        raise LogError(
            f"{rho_path}: expected 2 columns (depth_m, rho_kg_m3), got {rho_table.shape[1]}"
        )
    d_vp, vp = vp_table[:, 0], vp_table[:, 1]
    d_rho, rho = rho_table[:, 0], rho_table[:, 1]
    _validate_depth(d_vp, str(vp_path))
    _validate_depth(d_rho, str(rho_path))
    _validate_property(vp, "vp_m_s", str(vp_path))
    _validate_property(rho, "rho_kg_m3", str(rho_path))
    # nearest-neighbour match of every vp depth to a rho depth
    pos = np.searchsorted(d_rho, d_vp)
    left = np.clip(pos - 1, 0, d_rho.size - 1)
    right = np.clip(pos, 0, d_rho.size - 1)
    take_right = np.abs(d_vp - d_rho[right]) <= np.abs(d_vp - d_rho[left])
    match_idx = np.where(take_right, right, left)
    deviation = np.abs(d_vp - d_rho[match_idx])
    if np.any(deviation > depth_tolerance_m):
        worst = int(np.argmax(deviation))
        n_bad = int(np.count_nonzero(deviation > depth_tolerance_m))
        raise LogError(
            f"{source}: {n_bad} vp depth(s) have no rho depth within tolerance "
            f"{depth_tolerance_m} m; worst deviation {deviation[worst]:.3g} m at sample "
            f"{worst}: vp depth {d_vp[worst]:.6g} m vs nearest rho depth "
            f"{d_rho[match_idx[worst]]:.6g} m"
        )
    return Log(0.5 * (d_vp + d_rho[match_idx]), vp, rho[match_idx])


# ---------------------------------------------------------------------------
# Blocked models
# ---------------------------------------------------------------------------


def load_blocked_model(path: str | Path, *, name: str | None = None) -> LayeredModel:
    """Load a blocked model: ``depth_m`` is the layer top; last row = half-space.

    The canonical models in ``data/`` (single_interface, water_bottom,
    thick_5layer) use this format, which keeps layer boundaries exact (no
    interpolation smearing).

    Raises
    ------
    LogError
        On wrong column count, fewer than 2 rows, a first depth other than
        0 m, or invalid values.
    """
    source = str(path)
    table = _parse_numeric_table(Path(path).read_text(encoding="utf-8"), source)
    if table.shape[1] != 3:
        raise LogError(
            f"{source}: expected 3 columns (depth_m, vp_m_s, rho_kg_m3), got {table.shape[1]}"
        )
    if table.shape[0] < 2:
        raise LogError(f"{source}: blocked model needs at least 2 rows (one layer + half-space)")
    depth_m = table[:, 0]
    vp_m_s = table[:, 1]
    rho_kg_m3 = table[:, 2]
    _validate_depth(depth_m, source)
    _validate_property(vp_m_s, "vp_m_s", source)
    _validate_property(rho_kg_m3, "rho_kg_m3", source)
    if depth_m[0] != 0.0:
        raise LogError(
            f"{source}: blocked model must start at depth 0 m (surface), got {depth_m[0]}"
        )
    model_name = name if name is not None else Path(path).stem
    return LayeredModel(h_m=np.diff(depth_m), vp_m_s=vp_m_s, rho_kg_m3=rho_kg_m3, name=model_name)


# ---------------------------------------------------------------------------
# Blocking onto a uniform-TWT grid
# ---------------------------------------------------------------------------


def _interval_one_way_times(
    depth_m: NDArray[np.float64], vp_m_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Exact one-way time per sample interval for linearly interpolated velocity.

    For velocity linear between samples, the slowness integral has the closed
    form ``h / dv * ln(v2 / v1)``, with the constant-velocity limit ``h / v``
    used when ``dv`` is negligible.
    """
    h = depth_m[1:] - depth_m[:-1]
    dv = vp_m_s[1:] - vp_m_s[:-1]
    v1 = vp_m_s[:-1]
    v2 = vp_m_s[1:]
    times = np.empty_like(h)
    linear = np.abs(dv) > 1e-9 * np.abs(v1)
    times[~linear] = h[~linear] / v1[~linear]
    times[linear] = h[linear] / dv[linear] * np.log(v2[linear] / v1[linear])
    return times


def _mean_piecewise_linear(
    depth_m: NDArray[np.float64], values: NDArray[np.float64], top_m: float, bottom_m: float
) -> float:
    """Exact mean of a piecewise-linear function of depth over ``[top, bottom]``."""
    inner = depth_m[(depth_m > top_m) & (depth_m < bottom_m)]
    nodes = np.concatenate(([top_m], inner, [bottom_m]))
    node_values = np.interp(nodes, depth_m, values)
    integral = np.sum(0.5 * (node_values[:-1] + node_values[1:]) * np.diff(nodes))
    return float(integral / (bottom_m - top_m))


def block_log_uniform_twt(
    log: Log,
    layer_twt_s: float,
    *,
    max_layers: int | None = None,
    name: str = "",
) -> LayeredModel:
    """Block a sampled log into a model with (near-)uniform layer two-way times.

    Layer boundaries are placed at multiples of ``layer_twt_s`` along the
    exact two-way-time axis (integrated slowness of the linearly interpolated
    velocity), so every layer has TWT ``layer_twt_s`` except the deepest,
    which absorbs any remainder (TWT between ``layer_twt_s`` and
    ``2 * layer_twt_s`` -- no sliver layers).  Layer velocity is the interval
    velocity ``h / dt`` (harmonic
    mean, consistent with the layer TWT); density is the exact arithmetic mean
    of the linearly interpolated density over the layer.  The whole logged
    interval is covered by the uniform cells; the half-space below duplicates
    the deepest cell, so the base interface has reflection coefficient
    exactly 0 (physically invisible) and the model preserves the log's total
    two-way time and mass exactly.

    Parameters
    ----------
    log : Log
        Validated sampled log.
    layer_twt_s : float
        Target two-way time thickness of each layer, s (> 0).
    max_layers : int, optional
        Cap on the number of finite layers; if the target TWT would exceed
        it, the layer TWT is increased so the model has exactly
        ``max_layers`` uniform finite layers (plus the half-space).
    name : str, optional
        Model name.

    Returns
    -------
    LayeredModel
        Blocked model with a zero-coefficient half-space base.
    """
    if not np.isfinite(layer_twt_s) or layer_twt_s <= 0.0:
        raise LogError(f"layer_twt_s must be > 0, got {layer_twt_s}")
    if max_layers is not None and max_layers < 1:
        raise LogError(f"max_layers must be >= 1, got {max_layers}")
    depth_m, vp_m_s, rho_kg_m3 = log.depth_m, log.vp_m_s, log.rho_kg_m3
    ow = np.concatenate(([0.0], np.cumsum(_interval_one_way_times(depth_m, vp_m_s))))
    ow_total = float(ow[-1])
    delta_ow = layer_twt_s / 2.0
    # n full-TWT layers; the deepest layer absorbs the remainder, so no layer
    # is thinner than layer_twt_s (no degenerate sliver cells)
    n_finite = max(int(np.floor(ow_total / delta_ow * (1.0 + 1e-9))), 1)
    if max_layers is not None and n_finite > max_layers:
        n_finite = int(max_layers)
        delta_ow = ow_total / n_finite
    edges_ow = np.concatenate((np.arange(n_finite) * delta_ow, [ow_total]))
    edges_depth = np.interp(edges_ow, ow, depth_m)
    h_m = np.diff(edges_depth)
    ow_layer = np.diff(edges_ow)
    vp_layer = h_m / ow_layer
    rho_layer = np.array(
        [
            _mean_piecewise_linear(depth_m, rho_kg_m3, edges_depth[k], edges_depth[k + 1])
            for k in range(n_finite)
        ]
    )
    # half-space below the log duplicates the deepest cell: the base interface
    # has R = 0 exactly and is invisible to the wavefield
    return LayeredModel(
        h_m=h_m,
        vp_m_s=np.append(vp_layer, vp_layer[-1]),
        rho_kg_m3=np.append(rho_layer, rho_layer[-1]),
        name=name,
    )
