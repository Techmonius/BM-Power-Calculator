"""Bending magnet power calculator package (Jupyter-focused).
Modules:
- magnets: magnet sets and utilities
- masks: rectangular mask and aperture definitions and angular window logic
- attenuation: NIST/XCOM attenuation via xraylib
- compute: core engine to compute screen power density
- ui_presets: defaults and initial test config
- ui_widgets: ipywidgets-based UI launcher
"""

__all__ = [
    "magnets",
    "masks",
    "attenuation",
    "compute",
    "ui_presets",
    "ui_widgets",
]
