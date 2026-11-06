"""Layered acoustic model and plane-wave reflection/transmission coefficients.

Conventions (AGENTS.md; single-sourced here and validated against analytic
anchors in ``tests/test_model.py``):

- Acoustic impedance ``Z = rho * vp`` in kg/(m^2 s).
- Interface reflection for a down-going pressure wave:
  ``R = (Z_below - Z_above) / (Z_below + Z_above)`` -- pressure convention,
  SEG-normal polarity (impedance increase gives a positive event).
- Pressure transmission at normal incidence:
  ``T_down = 2 Z_below / (Z_above + Z_below)`` and its upward analog
  ``T_up = 2 Z_above / (Z_above + Z_below)``; the round-trip product satisfies
  ``T_down * T_up = 1 - R**2`` at every angle.
- Angle dependence is parameterized by the horizontal slowness
  ``p = sin(theta)/vp`` in s/m -- the natural variable of the omega-p engine.
  With direction cosines ``c_i = sqrt(1 - (p vp_i)**2)``
  (complex-safe in the evanescent region),
  ``R(p) = (Z_b c_a - Z_a c_b) / (Z_b c_a + Z_a c_b)`` where indices ``a``/``b``
  denote the layers above/below the interface.
- Energy-flux-normalized transmission ``T_hat`` satisfies
  ``|R|**2 + |T_hat|**2 = 1`` at every angle for lossless media.  Past the
  critical angle (``p > 1/vp_below``, only possible when ``vp_below >
  vp_above``) the transmitted wave is evanescent: ``|R| = 1`` and ``T_hat = 0``.
- Angles are degrees at the public API, measured from the layer normal in the
  layer *above* the interface; radians live only inside computations.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mm1d.units import deg_to_rad

__all__ = [
    "Angle",
    "Coefficient",
    "LayeredModel",
    "ModelError",
    "Slowness",
    "acoustic_reflection_angle",
    "acoustic_reflection_p",
    "acoustic_transmission_angle",
    "acoustic_transmission_p",
    "energy_transmission_angle",
    "energy_transmission_p",
]

Angle = float | NDArray[np.float64]
"""Public-API angle in degrees: scalar or 1-D array."""

Slowness = float | NDArray[np.float64]
"""Horizontal slowness p in s/m: scalar or 1-D array."""

Coefficient = float | complex | NDArray[np.float64] | NDArray[np.complex128]
"""Reflection/transmission coefficient: real while subcritical, complex past critical."""


class ModelError(ValueError):
    """Raised when a layered model is inconsistent or a request is out of range."""


# ---------------------------------------------------------------------------
# Plane-wave interface coefficients (free functions, engine-facing)
# ---------------------------------------------------------------------------


def _interface_optics(
    vp_above_m_s: float,
    rho_above_kg_m3: float,
    vp_below_m_s: float,
    rho_below_kg_m3: float,
    p_s_m: Slowness,
) -> tuple[float, float, NDArray[np.float64], NDArray | NDArray[np.complex128], bool]:
    """Impedances and direction cosines for a ray parameter p.

    Returns ``(Z_above, Z_below, c_above, c_below, scalar)`` with
    ``c_i = sqrt(1 - (p vp_i)**2)``; ``c_below`` is complex-safe (imaginary in
    the evanescent region ``p > 1/vp_below``).
    """
    p_arr = np.asarray(p_s_m, dtype=np.float64)
    scalar = p_arr.ndim == 0
    p_arr = np.atleast_1d(p_arr)
    if np.any(p_arr * vp_above_m_s > 1.0):
        raise ModelError(
            f"horizontal slowness p exceeds the incidence limit 1/vp_above = "
            f"{1.0 / vp_above_m_s:.6g} s/m: no propagating incident wave in the "
            f"layer above the interface"
        )
    c_above = np.sqrt(1.0 - (p_arr * vp_above_m_s) ** 2)
    c_below = np.emath.sqrt(1.0 - (p_arr * vp_below_m_s) ** 2)
    return rho_above_kg_m3 * vp_above_m_s, rho_below_kg_m3 * vp_below_m_s, c_above, c_below, scalar


def _maybe_scalar(values: NDArray, scalar: bool) -> Coefficient:
    """Return a Python scalar for scalar input, the array otherwise."""
    if scalar:
        return values.reshape(-1)[0].item()
    return values


def acoustic_reflection_p(
    vp_above_m_s: float,
    rho_above_kg_m3: float,
    vp_below_m_s: float,
    rho_below_kg_m3: float,
    p_s_m: Slowness,
) -> Coefficient:
    """Pressure reflection coefficient R as a function of horizontal slowness.

    ``R(p) = (Z_b c_a - Z_a c_b) / (Z_b c_a + Z_a c_b)`` with
    ``c_i = sqrt(1 - (p v_i)**2)``.  Requires ``p <= 1/vp_above`` (propagating
    incident wave).  For ``1/vp_below < p <= 1/vp_above`` (total internal
    reflection) the result is complex with ``|R| = 1``.

    Parameters
    ----------
    vp_above_m_s, vp_below_m_s : float
        P-wave velocities of the layers above/below the interface, m/s.
    rho_above_kg_m3, rho_below_kg_m3 : float
        Densities of the layers above/below the interface, kg/m^3.
    p_s_m : float or ndarray
        Horizontal slowness, s/m.

    Returns
    -------
    Coefficient
        R(p) (dimensionless), real or complex (see above).
    """
    z_a, z_b, c_a, c_b, scalar = _interface_optics(
        vp_above_m_s, rho_above_kg_m3, vp_below_m_s, rho_below_kg_m3, p_s_m
    )
    r = (z_b * c_a - z_a * c_b) / (z_b * c_a + z_a * c_b)
    return _maybe_scalar(r, scalar)


def acoustic_transmission_p(
    vp_above_m_s: float,
    rho_above_kg_m3: float,
    vp_below_m_s: float,
    rho_below_kg_m3: float,
    p_s_m: Slowness,
) -> tuple[Coefficient, Coefficient]:
    """Pressure transmission coefficients (down, up) as a function of slowness.

    ``T_down = 2 Z_b c_a / (Z_b c_a + Z_a c_b)`` for a down-going wave and
    ``T_up = 2 Z_a c_b / (Z_b c_a + Z_a c_b)`` for the upward analog, so that
    ``T_down * T_up = 1 - R(p)**2`` at every slowness.

    Parameters are as in :func:`acoustic_reflection_p`.

    Returns
    -------
    tuple of Coefficient
        ``(T_down, T_up)``, dimensionless.
    """
    z_a, z_b, c_a, c_b, scalar = _interface_optics(
        vp_above_m_s, rho_above_kg_m3, vp_below_m_s, rho_below_kg_m3, p_s_m
    )
    denom = z_b * c_a + z_a * c_b
    t_down = 2.0 * z_b * c_a / denom
    t_up = 2.0 * z_a * c_b / denom
    return _maybe_scalar(t_down, scalar), _maybe_scalar(t_up, scalar)


def energy_transmission_p(
    vp_above_m_s: float,
    rho_above_kg_m3: float,
    vp_below_m_s: float,
    rho_below_kg_m3: float,
    p_s_m: Slowness,
) -> tuple[Coefficient, Coefficient]:
    """Energy-flux-normalized transmission coefficients (down, up) vs slowness.

    ``T_hat_down = T_down * sqrt(Z_a c_b_flux / (Z_b c_a))`` with
    ``c_b_flux = Re(c_b)`` (zero in the evanescent region), and the mirrored
    expression for ``T_hat_up``.  They satisfy ``|R|**2 + |T_hat|**2 = 1`` for
    lossless media at every slowness; both are exactly 0 past the critical
    slowness ``p = 1/vp_below`` (no vertical energy flux transmitted).

    Parameters are as in :func:`acoustic_reflection_p`.

    Returns
    -------
    tuple of Coefficient
        ``(T_hat_down, T_hat_up)``, dimensionless, real-valued.
    """
    z_a, z_b, c_a, c_b, scalar = _interface_optics(
        vp_above_m_s, rho_above_kg_m3, vp_below_m_s, rho_below_kg_m3, p_s_m
    )
    denom = z_b * c_a + z_a * c_b
    t_down = 2.0 * z_b * c_a / denom
    t_up = 2.0 * z_a * c_b / denom
    c_b_flux = np.real(c_b)  # equals c_b while propagating, 0 while evanescent
    propagating = (c_b_flux > 0.0) & (c_a > 0.0)
    flux_ratio_down = np.zeros_like(c_a)
    flux_ratio_up = np.zeros_like(c_a)
    flux_ratio_down[propagating] = (z_a * c_b_flux[propagating]) / (z_b * c_a[propagating])
    flux_ratio_up[propagating] = (z_b * c_a[propagating]) / (z_a * c_b_flux[propagating])
    t_hat_down = np.real_if_close(t_down * np.sqrt(flux_ratio_down))
    t_hat_up = np.real_if_close(t_up * np.sqrt(flux_ratio_up))
    return _maybe_scalar(t_hat_down, scalar), _maybe_scalar(t_hat_up, scalar)


def acoustic_reflection_angle(
    vp_above_m_s: float,
    rho_above_kg_m3: float,
    vp_below_m_s: float,
    rho_below_kg_m3: float,
    angle_above_deg: Angle,
) -> Coefficient:
    """Pressure reflection coefficient R(theta); angle in the layer above, degrees."""
    p = np.sin(deg_to_rad(angle_above_deg)) / vp_above_m_s
    return acoustic_reflection_p(vp_above_m_s, rho_above_kg_m3, vp_below_m_s, rho_below_kg_m3, p)


def acoustic_transmission_angle(
    vp_above_m_s: float,
    rho_above_kg_m3: float,
    vp_below_m_s: float,
    rho_below_kg_m3: float,
    angle_above_deg: Angle,
) -> tuple[Coefficient, Coefficient]:
    """Pressure transmission coefficients (down, up) vs angle; degrees in layer above."""
    p = np.sin(deg_to_rad(angle_above_deg)) / vp_above_m_s
    return acoustic_transmission_p(vp_above_m_s, rho_above_kg_m3, vp_below_m_s, rho_below_kg_m3, p)


def energy_transmission_angle(
    vp_above_m_s: float,
    rho_above_kg_m3: float,
    vp_below_m_s: float,
    rho_below_kg_m3: float,
    angle_above_deg: Angle,
) -> tuple[Coefficient, Coefficient]:
    """Energy-flux-normalized transmission (down, up) vs angle; degrees in layer above."""
    p = np.sin(deg_to_rad(angle_above_deg)) / vp_above_m_s
    return energy_transmission_p(vp_above_m_s, rho_above_kg_m3, vp_below_m_s, rho_below_kg_m3, p)


# ---------------------------------------------------------------------------
# Layered model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class LayeredModel:
    """Horizontally layered acoustic medium; the last layer is a half-space.

    Attributes
    ----------
    h_m : ndarray
        Thicknesses of the finite (non-half-space) layers, m; shape
        ``(n_layers - 1,)``.
    vp_m_s : ndarray
        P-wave velocity of every layer including the half-space, m/s; shape
        ``(n_layers,)``.
    rho_kg_m3 : ndarray
        Density of every layer including the half-space, kg/m^3.
    name : str
        Label used in plots and reports.

    Notes
    -----
    Interface ``i`` lies between layers ``i`` (above) and ``i + 1`` (below);
    valid indices are ``0 .. n_interfaces - 1`` with
    ``n_interfaces = n_layers - 1``.
    """

    h_m: NDArray[np.float64]
    vp_m_s: NDArray[np.float64]
    rho_kg_m3: NDArray[np.float64]
    name: str = ""

    def __post_init__(self) -> None:
        """Copy inputs to float64 arrays and validate the layer stack."""
        h = np.array(self.h_m, dtype=np.float64, copy=True)
        vp = np.array(self.vp_m_s, dtype=np.float64, copy=True)
        rho = np.array(self.rho_kg_m3, dtype=np.float64, copy=True)
        for label, arr in (("h_m", h), ("vp_m_s", vp), ("rho_kg_m3", rho)):
            if arr.ndim != 1:
                raise ModelError(f"{label} must be a 1-D array, got shape {arr.shape}")
        n = vp.size
        if n < 2:
            raise ModelError(f"need at least 2 layers (finite layers + half-space), got {n}")
        if rho.size != n:
            raise ModelError(f"vp_m_s and rho_kg_m3 must have equal length, got {n} and {rho.size}")
        if h.size != n - 1:
            raise ModelError(
                f"h_m must have length n_layers - 1 = {n - 1} (the last layer is the "
                f"half-space), got {h.size}"
            )
        for label, arr in (("h_m", h), ("vp_m_s", vp), ("rho_kg_m3", rho)):
            bad = np.flatnonzero(~np.isfinite(arr))
            if bad.size:
                raise ModelError(
                    f"{label} contains non-finite value(s) at index {bad[:5].tolist()}"
                )
        bad = np.flatnonzero(h <= 0.0)
        if bad.size:
            raise ModelError(f"layer thicknesses must be > 0 m; violation at {bad[:5].tolist()}")
        for label, arr in (("vp_m_s", vp), ("rho_kg_m3", rho)):
            bad = np.flatnonzero(arr <= 0.0)
            if bad.size:
                raise ModelError(f"{label} must be > 0; violation at {bad[:5].tolist()}")
        object.__setattr__(self, "h_m", h)
        object.__setattr__(self, "vp_m_s", vp)
        object.__setattr__(self, "rho_kg_m3", rho)

    def __repr__(self) -> str:
        """Short one-line representation."""
        return (
            f"LayeredModel(name={self.name!r}, n_layers={self.n_layers}, "
            f"total_depth_m={self.total_depth_m:.1f})"
        )

    # -- geometry -----------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Number of layers, including the half-space."""
        return self.vp_m_s.size

    @property
    def n_interfaces(self) -> int:
        """Number of interfaces (every finite layer has one at its base)."""
        return self.n_layers - 1

    @property
    def impedance(self) -> NDArray[np.float64]:
        """Acoustic impedance ``rho * vp`` per layer, kg/(m^2 s)."""
        return self.rho_kg_m3 * self.vp_m_s

    @property
    def interface_depth_m(self) -> NDArray[np.float64]:
        """Depth of each interface (base of each finite layer), m."""
        return np.cumsum(self.h_m)

    @property
    def total_depth_m(self) -> float:
        """Depth of the top of the half-space, m."""
        return float(self.h_m.sum())

    @property
    def layer_twt_s(self) -> NDArray[np.float64]:
        """Two-way time thickness of each finite layer, s (``2 h / vp``)."""
        return 2.0 * self.h_m / self.vp_m_s[:-1]

    @property
    def interface_twt_s(self) -> NDArray[np.float64]:
        """Zero-offset two-way time from the surface to each interface, s."""
        return np.cumsum(self.layer_twt_s)

    # -- coefficients --------------------------------------------------------

    @property
    def reflections_normal(self) -> NDArray[np.float64]:
        """Normal-incidence reflection coefficient of every interface."""
        z = self.impedance
        return (z[1:] - z[:-1]) / (z[1:] + z[:-1])

    @property
    def transmissions_normal(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normal-incidence pressure transmission ``(T_down, T_up)`` per interface.

        ``T_down = 2 Z_below / (Z_above + Z_below)`` for a down-going wave and
        ``T_up = 2 Z_above / (Z_above + Z_below)`` for the upward analog, so the
        round-trip product satisfies ``T_down * T_up = 1 - R**2`` for every
        interface.
        """
        z = self.impedance
        total = z[1:] + z[:-1]
        return 2.0 * z[1:] / total, 2.0 * z[:-1] / total

    def _check_interface(self, i: int) -> None:
        """Raise ModelError unless ``i`` is a valid interface index."""
        if not 0 <= i < self.n_interfaces:
            raise ModelError(
                f"interface index {i} out of range [0, {self.n_interfaces - 1}] "
                f"for a model with {self.n_layers} layers"
            )

    def reflection(self, i: int, angle_deg: Angle = 0.0) -> Coefficient:
        """Reflection coefficient of interface ``i`` at normal incidence or angle.

        Parameters
        ----------
        i : int
            Interface index (0 = base of the top layer).
        angle_deg : float or ndarray, optional
            Incidence angle in the layer above the interface, degrees
            (0 by default = normal incidence).

        Returns
        -------
        Coefficient
            R, dimensionless; complex with |R| = 1 past the critical angle.
        """
        self._check_interface(i)
        return acoustic_reflection_angle(
            self.vp_m_s[i], self.rho_kg_m3[i], self.vp_m_s[i + 1], self.rho_kg_m3[i + 1], angle_deg
        )

    def transmission(self, i: int, angle_deg: Angle = 0.0) -> tuple[Coefficient, Coefficient]:
        """Pressure transmission ``(T_down, T_up)`` of interface ``i``.

        Parameters are as in :meth:`reflection`.
        """
        self._check_interface(i)
        return acoustic_transmission_angle(
            self.vp_m_s[i], self.rho_kg_m3[i], self.vp_m_s[i + 1], self.rho_kg_m3[i + 1], angle_deg
        )

    def energy_transmission(
        self, i: int, angle_deg: Angle = 0.0
    ) -> tuple[Coefficient, Coefficient]:
        """Energy-flux-normalized transmission ``(T_hat_down, T_hat_up)``.

        Satisfies ``|R|^2 + |T_hat|^2 = 1`` (see module docstring).
        Parameters are as in :meth:`reflection`.
        """
        self._check_interface(i)
        return energy_transmission_angle(
            self.vp_m_s[i], self.rho_kg_m3[i], self.vp_m_s[i + 1], self.rho_kg_m3[i + 1], angle_deg
        )

    def critical_angle_deg(self, i: int) -> float | None:
        """Critical angle of interface ``i`` for incidence from above, degrees.

        None when the velocity decreases across the interface (no critical
        angle exists in that case).
        """
        self._check_interface(i)
        vp_above, vp_below = self.vp_m_s[i], self.vp_m_s[i + 1]
        if vp_below <= vp_above:
            return None
        return math.degrees(math.asin(vp_above / vp_below))

    def summary(self) -> str:
        """Return a human-readable one-line-per-layer description of the model."""
        lines = [
            f"LayeredModel {self.name!r}: {self.n_layers} layers "
            f"(total depth {self.total_depth_m:.1f} m, "
            f"TWT to base {self.interface_twt_s[-1]:.3f} s)",
            "  #      h_m    vp_m_s  rho_kg_m3   Z_kg_m2_s   twt_s       R",
        ]
        z = self.impedance
        twt = self.layer_twt_s
        refl = self.reflections_normal
        for i in range(self.n_layers - 1):
            lines.append(
                f"{i:3d} {self.h_m[i]:8.1f} {self.vp_m_s[i]:9.1f} {self.rho_kg_m3[i]:9.1f} "
                f"{z[i]:11.1f} {twt[i]:8.3f} {refl[i]:+8.4f}"
            )
        lines.append(
            f"{self.n_layers - 1:3d} {'inf':>8} {self.vp_m_s[-1]:9.1f} "
            f"{self.rho_kg_m3[-1]:9.1f} {z[-1]:11.1f} {'-':>8} {'-':>8}"
        )
        return "\n".join(lines)
