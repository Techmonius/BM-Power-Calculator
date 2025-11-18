from __future__ import annotations

from typing import List

from .masks import Mask, Aperture
from .attenuation import Attenuator, DEFAULT_DENSITIES
from .compute import Settings


def materials_dropdown_options() -> List[str]:
    """
    Return the list of elemental materials available for attenuators.
    """
    return list(DEFAULT_DENSITIES.keys())


def default_settings() -> Settings:
    """
    Default computation settings per your specification.
    """
    return Settings(
        distances_unit="mm",
        energy_GeV=6.0,
        current_A=0.22,
        screen_z_mm=7000.0,  # can be changed in UI; set downstream of attenuator/mask for demo
        nx=1024,
        ny=1024,
        Emin_eV=100.0,
        Emax_eV=50000.0,
        nE=1000,
        logE=True,
        vert_sigma_multiplier=5.0,
    )


def default_masks() -> List[Mask]:
    """
    Preload your test mask configuration:
    Mask at z = 6269 mm with 4 apertures
      - [1,5] at [-3.15, 0]
      - [1,5] at [-6.92, 0]
      - [1.5,5] at [-10.7, 0]
      - [1,5] at [-14.42, 0]
    """
    apertures = [
        Aperture(width_mm=1.0, height_mm=5.0, center_x_mm=-3.15, center_y_mm=0.0),
        Aperture(width_mm=1.0, height_mm=5.0, center_x_mm=-6.92, center_y_mm=0.0),
        Aperture(width_mm=1.5, height_mm=5.0, center_x_mm=-10.7, center_y_mm=0.0),
        Aperture(width_mm=1.0, height_mm=5.0, center_x_mm=-14.42, center_y_mm=0.0),
    ]
    return [Mask(name="Mask6269", z_mm=6269.0, apertures=apertures)]


def default_attenuators() -> List[Attenuator]:
    """
    Preload your test attenuator configuration:
      - Attenuator at z = 5000 mm, Beryllium (Be), thickness 0.25 mm
    """
    return [Attenuator(z_mm=5000.0, material="Be", thickness_mm=0.25)]


def default_magnet_set_title() -> str:
    """
    Default magnet set selection title.
    """
    return "BM (M3, Q8, M4)"
