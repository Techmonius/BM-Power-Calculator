import numpy as np

from src.masks import Mask, Aperture, mask_angular_windows, combined_angular_windows_upstream
from src.compute import Settings, compute_screen_power
from src.magnets import get_available_magnet_sets
from src.footprints import extract_footprints


def test_mask_angular_windows_simple():
    # Single mask, single aperture, easy small-angle geometry
    z_mm = 1000.0
    ap = Aperture(width_mm=2.0, height_mm=4.0, center_x_mm=0.0, center_y_mm=0.0)
    m = Mask(name="M", z_mm=z_mm, apertures=[ap])

    tx, ty = mask_angular_windows(m)
    assert len(tx) == 1 and len(ty) == 1

    # Expected: theta_x in [-1e-3, +1e-3], theta_y in [-2e-3, +2e-3]
    assert np.isclose(tx[0][0], -1.0e-3, atol=1e-9)
    assert np.isclose(tx[0][1], +1.0e-3, atol=1e-9)
    assert np.isclose(ty[0][0], -2.0e-3, atol=1e-9)
    assert np.isclose(ty[0][1], +2.0e-3, atol=1e-9)


def test_combined_angular_windows_intersection():
    # Two masks with partial horizontal overlap:
    # m1: at z=1000 mm with x in [-1, +1] mm -> theta in [-1e-3, +1e-3]
    # m2: at z=2000 mm with x in [+1, +3] mm -> theta in [+0.5e-3, +1.5e-3]
    # Intersection across masks should be [0.5e-3, 1.0e-3]
    m1 = Mask(
        name="M1",
        z_mm=1000.0,
        apertures=[Aperture(width_mm=2.0, height_mm=10.0, center_x_mm=0.0, center_y_mm=0.0)],
    )
    m2 = Mask(
        name="M2",
        z_mm=2000.0,
        apertures=[Aperture(width_mm=2.0, height_mm=10.0, center_x_mm=2.0, center_y_mm=0.0)],
    )
    tx, ty = combined_angular_windows_upstream([m1, m2], screen_z_mm=5000.0)
    assert len(tx) == 1
    assert np.isclose(tx[0][0], 0.5e-3, atol=5e-7)
    assert np.isclose(tx[0][1], 1.0e-3, atol=5e-7)
    # Vertical windows non-empty
    assert isinstance(ty, list)


def test_compute_runs_and_shapes():
    # Use a predefined magnet set and a simple upstream mask
    msets = get_available_magnet_sets()
    # pick any available set deterministically
    mset = msets[sorted(msets.keys())[0]]

    s = Settings(
        distances_unit="mm",
        energy_GeV=6.0,
        current_A=0.22,
        screen_z_mm=7000.0,
        nx=64,
        ny=32,
        Emin_eV=100.0,
        Emax_eV=1000.0,
        nE=50,
        logE=True,
        vert_sigma_multiplier=5.0,
        do_validation=False,  # keep fast
    )
    masks = [Mask(name="A", z_mm=6000.0, apertures=[Aperture(width_mm=1.0, height_mm=5.0, center_x_mm=0.0, center_y_mm=0.0)])]
    atts = []  # keep empty to avoid xraylib

    res = compute_screen_power(mset, masks, atts, s)

    # Shapes and basic properties
    assert res.x_mm.shape == (s.nx,)
    assert res.y_mm.shape == (s.ny,)
    assert res.power_W_m2.shape == (s.ny, s.nx)
    assert np.all(res.power_W_m2 >= 0.0)
    assert res.allowed_count is not None and res.allowed_count > 0
    assert res.total_power_W >= 0.0
    # Line powers and theta arrays consistent
    assert res.line_power_x_W_per_mm.shape == (s.nx,)
    assert res.line_power_y_W_per_mm.shape == (s.ny,)
    assert res.theta_x_mrad.shape == (s.nx,)
    assert res.line_power_theta_W_per_mrad.shape == (s.nx,)


def test_footprints_extraction_basic():
    # Build a scenario with a single narrow aperture; expect at least one footprint
    msets = get_available_magnet_sets()
    mset = msets[sorted(msets.keys())[0]]

    z_screen = 7000.0
    z_mask = 6000.0
    w_mm = 1.0
    h_mm = 5.0

    s = Settings(
        distances_unit="mm",
        energy_GeV=6.0,
        current_A=0.22,
        screen_z_mm=z_screen,
        nx=128,
        ny=64,
        Emin_eV=100.0,
        Emax_eV=1000.0,
        nE=50,
        logE=True,
        vert_sigma_multiplier=5.0,
        do_validation=False,
    )
    masks = [Mask(name="M", z_mm=z_mask, apertures=[Aperture(width_mm=w_mm, height_mm=h_mm, center_x_mm=0.0, center_y_mm=0.0)])]
    atts = []

    res = compute_screen_power(mset, masks, atts, s)
    fps = extract_footprints(res, screen_z_mm=z_screen, rel_threshold=0.01, min_pixels=5, connectivity=1)

    assert isinstance(fps, list)
    assert len(fps) >= 1
    # Check first footprint has positive measures and power
    fp = fps[0]
    assert fp.width_x_mm >= 0.0 and fp.height_y_mm >= 0.0
    assert fp.total_power_W >= 0.0

    # The angular width (mrad) of the footprint should be close to aperture angular width w/z_mask (in mrad)
    expected_theta_mrad = (w_mm / z_mask) * 1e3
    assert np.isclose(fp.width_theta_mrad, expected_theta_mrad, rtol=0.15, atol=0.02)