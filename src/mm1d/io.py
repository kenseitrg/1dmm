"""CSV/TXT log loading, resampling, blocking.

Loaders accept either a single file with columns ``depth_m, vp_m_s, rho_kg_m3``
or two two-column files (``depth_m, vp_m_s`` and ``depth_m, rho_kg_m3``)
merged on depth (with tolerance). Validation: monotonic depth, positive
velocity/density, no NaNs — failures are loud with clear messages (M1+).
"""
