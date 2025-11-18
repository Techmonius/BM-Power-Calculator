from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Magnet:
    """
    Represents a single bending magnet source.

    Units:
      - B_T: Tesla
      - theta_min_mrad / theta_max_mrad: milliradians (horizontal fan acceptance)
      - z_mm: millimeters (position along beamline; 0 is the source point for the set)
      - x0_mm, y0_mm: millimeters (source center offsets; default to 0)
    """
    name: str
    B_T: float
    theta_min_mrad: float
    theta_max_mrad: float
    z_mm: float
    x0_mm: float = 0.0
    y0_mm: float = 0.0
    # Weight factor for contribution scaling if needed later
    weight: float = 1.0


@dataclass
class MagnetSet:
    """
    Represents a set of bending magnets that contribute additively.

    All magnets in a set share the same beamline axis; Z=0 is the source point
    of the set (M3 for BM set, A:M1 for ID set). Negative Z are upstream.
    """
    title: str
    magnets: List[Magnet] = field(default_factory=list)


def get_available_magnet_sets() -> Dict[str, MagnetSet]:
    """
    Returns the predefined magnet sets based on user-provided data.
    """
    bm_set = MagnetSet(
        title="BM (M3, Q8, M4)",
        magnets=[
            Magnet(name="M3", B_T=0.653, theta_min_mrad=-2.8, theta_max_mrad=+2.8, z_mm=0.0),
            Magnet(name="Q8", B_T=0.186, theta_min_mrad=0.0, theta_max_mrad=+2.8, z_mm=-1100.0),
            Magnet(name="M4", B_T=0.609, theta_min_mrad=0.0, theta_max_mrad=+2.8, z_mm=-1422.0),
        ],
    )

    id_set = MagnetSet(
        title="ID (A:M1, B:M1)",
        magnets=[
            Magnet(name="A:M1", B_T=0.650, theta_min_mrad=-2.8, theta_max_mrad=0.0, z_mm=0.0),
            Magnet(name="B:M1", B_T=0.650, theta_min_mrad=0.0, theta_max_mrad=+2.8, z_mm=-7676.6),
        ],
    )

    return {
        bm_set.title: bm_set,
        id_set.title: id_set,
    }


def mrad_to_rad(mrad: float) -> float:
    return mrad * 1e-3


def rad_to_mrad(rad: float) -> float:
    return rad * 1e3