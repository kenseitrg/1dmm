"""Layered model and reflection/transmission coefficients.

``LayeredModel`` holds per-layer thickness, P-velocity and density, plus
interface reflection/transmission coefficients at normal incidence and the
angle-dependent acoustic R(theta), T(theta). Sign convention (AGENTS.md):
``R = (Z_below - Z_above) / (Z_below + Z_above)`` (pressure, SEG-normal
polarity), round-trip transmission ``T_down * T_up = 1 - R**2``.
"""
