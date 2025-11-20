from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math

try:
    import xraylib
except Exception:  # pragma: no cover - optional at import time
    xraylib = None


# Default elemental densities in g/cm^3
DEFAULT_DENSITIES: Dict[str, float] = {
    "Be": 1.848,  # Beryllium
    "Al": 2.70,
    "Cu": 8.96,
    "Si": 2.33,
    "W": 19.25,
    "C": 2.267,  # graphite
}

# Periodic table Z numbers for common elements we support initially
ELEMENT_Z: Dict[str, int] = {
    "H": 1,
    "Be": 4,
    "C": 6,
    "Al": 13,
    "Si": 14,
    "Cu": 29,
    "W": 74,
}


@dataclass
class Attenuator:
    """
    Single attenuator layer placed at a Z position.

    Units:
      - z_mm: millimeters (position along beamline)
      - thickness_mm: millimeters
      - material: element symbol (e.g., 'Be', 'Al')
      - density_g_cm3: if None, use DEFAULT_DENSITIES
    """
    z_mm: float
    material: str
    thickness_mm: float
    density_g_cm3: float | None = None

    def density(self) -> float:
        if self.density_g_cm3 is not None:
            return self.density_g_cm3
        if self.material not in DEFAULT_DENSITIES:
            raise ValueError(f"Unknown density for material {self.material}; please provide explicitly.")
        return DEFAULT_DENSITIES[self.material]

    def Z(self) -> int:
        if self.material not in ELEMENT_Z:
            raise ValueError(f"Unsupported element {self.material}; add Z mapping in ELEMENT_Z.")
        return ELEMENT_Z[self.material]


def transmission_curve(attenuators: List[Attenuator], energies_eV: List[float]) -> List[float]:
    """
    Compute the total transmission curve T(E) across all attenuators (upstream of the screen),
    using NIST/XCOM mass attenuation coefficients via xraylib.

    Inputs:
      - attenuators: list of Attenuator
      - energies_eV: list or array of energies in eV

    Returns:
      - T_E: list of transmission values per energy (unitless, 0..1)
    """
    if xraylib is None:
        # Fallback when xraylib is not available: no attenuation applied
        # This allows the rest of the computation to proceed.
        print("WARNING: Attenuation is disabled because xraylib is not installed. Transmission set to 1.0 across all energies.")
        return [1.0 for _ in energies_eV]

    # xraylib expects energies in keV
    energies_keV = [e / 1e3 for e in energies_eV]

    # Initialize ln(T) accumulator for numerical stability
    lnT = [0.0 for _ in energies_keV]

    for att in attenuators:
        Z = att.Z()
        rho = att.density()  # g/cm^3
        t_cm = att.thickness_mm / 10.0  # convert mm -> cm
        if t_cm <= 0:
            continue
        # mass attenuation coeff mu_over_rho(E) in cm^2/g
        mu_over_rho = [xraylib.CS_Total(Z, Ek) for Ek in energies_keV]
        # mu(E) = (mu/rho)*rho in 1/cm
        mu = [m * rho for m in mu_over_rho]
        # contribution to ln(T) is -mu * t
        for i, mui in enumerate(mu):
            lnT[i] += -mui * t_cm

    return [math.exp(v) for v in lnT]


def filter_upstream(attenuators: List[Attenuator], screen_z_mm: float) -> List[Attenuator]:
    """
    Return only attenuators with z_mm < screen_z_mm.
    """
    return [a for a in attenuators if a.z_mm < screen_z_mm]
