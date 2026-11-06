"""Unit constants and conversions (single source of truth).

Internally the library uses SI everywhere: depth ``m``, layer thickness ``m``,
time ``s``, frequency ``Hz``, velocity ``m/s``, density ``kg/m³``,
slowness ``s/m``. Angles are degrees at the public API and radians internally;
any conversion happens exactly once, at the boundary.

This module will hold those constants and conversion helpers (M1+).
"""
