"""mm1d: 1.5D reflectivity synthetic modelling for multiple analysis.

Computes synthetic seismic wavefields from P-velocity and density well logs for
a horizontally layered (1.5D) earth, in three multiple-content variants:

- ``"full"``: primaries + surface-related multiples + interbed multiples
- ``"no_surface"``: primaries + interbed multiples (no free-surface reflections)
- ``"primaries"``: single-scattering response only

Pipeline (one stage per module group): ``logs → model → reflectivity →
synthetics → gathers → semblance → events/viz``.
"""

from mm1d.model import LayeredModel
from mm1d.reflectivity import ALL_VARIANTS, WavefieldVariant

__version__ = "0.1.0"

__all__ = ["ALL_VARIANTS", "LayeredModel", "WavefieldVariant", "__version__"]
