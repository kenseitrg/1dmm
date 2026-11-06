"""Zero-offset, angle gathers, and angle-to-offset conversion.

Angle gathers are assembled per wavefield variant over theta in [0, theta_max];
offset gathers follow via Snell's-law ray tracing: ``x = Σ h_i tan(theta_i)``,
"T = Σ h_i / (v_i cos(theta_i))" per leg (M3-M4).
"""
