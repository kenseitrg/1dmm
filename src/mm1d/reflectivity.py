"""ω-p reflectivity recursion and wavefield variants.

Recursive (compound-matrix style, numerically stable) computation of stack
reflectivity/transmissivity in the frequency-slowness domain per Müller (1985),
with the three wavefield variants selected by boundary/coefficient switches:
free-surface reflector on/off and interbed reverberation on/off.
"""

from typing import Literal

__all__ = ["ALL_VARIANTS", "WavefieldVariant"]

WavefieldVariant = Literal["full", "no_surface", "primaries"]
"""Wavefield variant name (single source of truth for the three variants).

- ``"full"``: primaries + surface-related + interbed multiples
- ``"no_surface"``: primaries + interbed multiples, no free-surface reflections
- ``"primaries"``: primaries only (single scattering)
"""

ALL_VARIANTS: tuple[WavefieldVariant, ...] = ("full", "no_surface", "primaries")
"""Canonical ordering of the three wavefield variants."""
