from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from .magnets import Magnet, MagnetSet, get_available_magnet_sets, mrad_to_rad
from .masks import Mask, combined_angular_windows_upstream
from .attenuation import Attenuator, filter_upstream, transmission_curve

# xrt imports
try:
    from xrt.backends import raycing
    from xrt.backends.raycing.sources import BendingMagnet as XrtBendingMagnet
except Exception as e:  # pragma: no cover
    XrtBendingMagnet = None


@dataclass
class Settings:
    distances_unit: str = "mm"  # fixed to mm per spec
    energy_GeV: float = 6.0
    current_A: float = 0.22
    screen_z_mm: float = 1000.0
    nx: int = 1024
    ny: int = 1024
    Emin_eV: float = 100.0
    Emax_eV: float = 50000.0
    nE: int = 1000
    logE: bool = True
    # vertical angular window multiplier when unconstrained by masks
    vert_sigma_multiplier: float = 5.0
    # Validation sampling for accurate vertical integration
    validate_theta_samples: int = 201
    validate_psi_samples: int = 51
    do_validation: bool = True
    # xrt Stokes S0 is often per 0.1% bandwidth; convert to per eV if True
    s0_is_per_0p1bw: bool = True


@dataclass
class ScreenResult:
    x_mm: np.ndarray  # shape (nx,)
    y_mm: np.ndarray  # shape (ny,)
    power_W_m2: np.ndarray  # shape (ny, nx)
    # Line power density at screen (integrate over the other axis):
    # Lx(x) = sum_y power_W_m2(y,x) * dy_mm (W/mm), Ly(y) = sum_x power_W_m2(y,x) * dx_mm (W/mm)
    line_power_x_W_per_mm: np.ndarray  # shape (nx,)
    line_power_y_W_per_mm: np.ndarray  # shape (ny,)
    # Horizontal angle grid and line power per mrad
    theta_x_mrad: np.ndarray  # shape (nx,)
    line_power_theta_W_per_mrad: np.ndarray  # shape (nx,)
    total_power_W: float
    energies_eV: np.ndarray  # shape (nE,)
    transmission_E: np.ndarray  # shape (nE,)
    theta_x_windows_rad: List[Tuple[float, float]]
    theta_y_windows_rad: List[Tuple[float, float]]
    # Validation results (optional)
    validation_total_W: Optional[float] = None
    validation_delta_percent: Optional[float] = None
    # Unmasked (no masks applied) validation total power (optional)
    unmasked_total_W: Optional[float] = None
    # Diagnostics: allowed mask area on the screen grid
    allowed_fraction_percent: Optional[float] = None
    allowed_count: Optional[int] = None
    # Angular-grid diagnostic total (integrate I(θ,ψ) over θ and ψ using the same grids/windows)
    angular_total_W: Optional[float] = None
    delta_screen_vs_angular_percent: Optional[float] = None
    delta_angular_vs_validation_percent: Optional[float] = None


def _energy_grid(settings: Settings) -> np.ndarray:
    if settings.logE:
        return np.logspace(np.log10(settings.Emin_eV), np.log10(settings.Emax_eV), settings.nE)
    return np.linspace(settings.Emin_eV, settings.Emax_eV, settings.nE)


def _gamma(energy_GeV: float) -> float:
    # gamma = 1957 * E (E in GeV)
    return 1957.0 * energy_GeV


def _sigma_vert_rad(energy_GeV: float) -> float:
    # sigma = 0.608/gamma = 0.311/E (E in GeV), in radians
    return 0.608 / _gamma(energy_GeV)


def _horizontal_theta_union_from_magnets(magnets: List[Magnet]) -> List[Tuple[float, float]]:
    # Union of horizontal theta ranges across magnets (in radians)
    intervals: List[Tuple[float, float]] = []
    for m in magnets:
        th_min = mrad_to_rad(m.theta_min_mrad)
        th_max = mrad_to_rad(m.theta_max_mrad)
        intervals.append((min(th_min, th_max), max(th_min, th_max)))
    # merge
    if not intervals:
        return []
    intervals.sort(key=lambda t: t[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged


def _auto_fov_mm(theta_x_windows: List[Tuple[float, float]], theta_y_windows: List[Tuple[float, float]],
                 settings: Settings) -> Tuple[np.ndarray, np.ndarray]:
    z = settings.screen_z_mm

    # Helper to check finiteness
    def _finite_windows(ws: List[Tuple[float, float]]):
        return [(a, b) for (a, b) in ws if math.isfinite(a) and math.isfinite(b)]

    tx = _finite_windows(theta_x_windows) if theta_x_windows else []
    ty = _finite_windows(theta_y_windows) if theta_y_windows else []

    # If windows are empty (blocked) or non-finite, return a tiny grid to avoid errors.
    if not tx or not ty:
        x = np.linspace(-1e-3, 1e-3, settings.nx)
        y = np.linspace(-1e-3, 1e-3, settings.ny)
        return x, y

    # Use union bounds for X and Y and create continuous grids covering the union span.
    tx_min = min(w[0] for w in tx)
    tx_max = max(w[1] for w in tx)
    ty_min = min(w[0] for w in ty)
    ty_max = max(w[1] for w in ty)

    x_min_mm = tx_min * z
    x_max_mm = tx_max * z
    y_min_mm = ty_min * z
    y_max_mm = ty_max * z

    # Add a slight margin (5%) to avoid clipping due to discretization
    mx = 0.05 * max(1.0, abs(x_max_mm - x_min_mm))
    my = 0.05 * max(1.0, abs(y_max_mm - y_min_mm))

    # Guard against any residual non-finite after math
    if not all(map(math.isfinite, [x_min_mm, x_max_mm, y_min_mm, y_max_mm])):
        x = np.linspace(-1e-3, 1e-3, settings.nx)
        y = np.linspace(-1e-3, 1e-3, settings.ny)
        return x, y

    x = np.linspace(x_min_mm - mx, x_max_mm + mx, settings.nx)
    y = np.linspace(y_min_mm - my, y_max_mm + my, settings.ny)
    return x, y


def _theta_grids_from_screen(x_mm: np.ndarray, y_mm: np.ndarray, z_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    # Small-angle approximation: theta = coord / z
    theta_x = x_mm / z_mm
    theta_y = y_mm / z_mm
    return theta_x, theta_y


def _apply_angular_windows_mask(theta_x: np.ndarray, theta_y: np.ndarray,
                                theta_x_windows: List[Tuple[float, float]],
                                theta_y_windows: List[Tuple[float, float]]) -> np.ndarray:
    # Build boolean mask for allowed angles based on windows
    mask_x = np.zeros_like(theta_x, dtype=bool)
    for s, e in theta_x_windows:
        mask_x |= (theta_x >= s) & (theta_x <= e)
    mask_y = np.zeros_like(theta_y, dtype=bool)
    for s, e in theta_y_windows:
        mask_y |= (theta_y >= s) & (theta_y <= e)
    # 2D combine via outer product
    return np.outer(mask_y, mask_x)


def _bm_horizontal_spectral_angle_density(magnet: Magnet, energies_eV: np.ndarray, theta_x_rad: np.ndarray,
                                           settings: Settings) -> np.ndarray:
    """
    Use xrt's BendingMagnet to compute spectral-angular density along horizontal for each energy.
    Returns array shape (nE, nThetaX) in W/sr/eV or consistent units as per xrt; we will integrate and scale.
    """
    if XrtBendingMagnet is None:
        print("WARNING: xrt is not installed. Using surrogate horizontal spectral profile (no absolute scaling).")
        nE = energies_eV.size
        nTx = theta_x_rad.size
        spectral = np.zeros((nE, nTx), dtype=float)
        gamma = _gamma(settings.energy_GeV)
        theta_c = 1.0 / gamma
        for iE in range(nE):
            spectral[iE, :] = 1.0 / (1.0 + (theta_x_rad / theta_c) ** 2) ** 2
        spectral *= magnet.B_T * settings.energy_GeV**3 * settings.current_A
        return spectral

    # xrt BM expects: eE (electron energy in GeV), I (A), B0 (T)
    # We compute at a far-field observation with given horizontal angles theta_x and vertical angle 0 (slice along horizontal),
    # then extend to 2D by multiplying with vertical Gaussian distribution as per provided sigma.

    # Create a BM source object. xrt's BM can be used to evaluate intensity via its methods.
    # Construct BM and set attributes to match your xrt version (no 'I' kwarg)
    bm = XrtBendingMagnet()
    try:
        bm.eE = settings.energy_GeV
    except Exception:
        pass
    try:
        bm.B0 = magnet.B_T
    except Exception:
        pass
    # Set current via an available attribute
    for _attr in ("I", "eI", "ringCurrent", "current"):
        if hasattr(bm, _attr):
            try:
                setattr(bm, _attr, settings.current_A)
                break
            except Exception:
                continue

    # xrt uses photon energy in eV
    # Prepare arrays for output
    nE = energies_eV.size
    nTx = theta_x_rad.size
    spectral = np.zeros((nE, nTx), dtype=float)

    # xrt raycing uses angles in radians; we'll sample vertical angle at 0 for horizontal profile.
    # NOTE: Depending on xrt's API, we might use bm.get_IvsTheta(E, thetaX, thetaY) or similar.
    # To keep it general, we approximate by evaluating differential power density via bm.radiation.
    # We'll fall back to a simplified universal function if a direct method isn't available.

    try:
        # Use xrt to compute Stokes parameters on a horizontal slice (psi=0) over energies and theta_x.
        psi = np.array([0.0], dtype=float)
        stokes = bm.intensities_on_mesh(energy=energies_eV, theta=theta_x_rad, psi=psi)
        # stokes is a list; S0 is total intensity with shape (nE, nTheta, nPsi[, harmonic])
        spectral = np.asarray(stokes[0])
        # squeeze the psi axis
        if spectral.ndim >= 3:
            spectral = np.squeeze(spectral, axis=2)
        # if there is still a harmonic axis, sum over it
        if spectral.ndim == 3:
            spectral = spectral.sum(axis=2)
    except Exception:
        print("WARNING: xrt call failed. Using surrogate horizontal spectral profile (no absolute scaling).")
        gamma = _gamma(settings.energy_GeV)
        theta_c = 1.0 / gamma
        for iE, _ in enumerate(energies_eV):
            spectral[iE, :] = 1.0 / (1.0 + (theta_x_rad / theta_c) ** 2) ** 2
        spectral *= magnet.B_T * settings.energy_GeV**3 * settings.current_A

    return spectral


def _vertical_profile(theta_y_rad: np.ndarray, settings: Settings) -> np.ndarray:
    # Gaussian with sigma = 0.608/gamma
    sigma = _sigma_vert_rad(settings.energy_GeV)
    return np.exp(-0.5 * (theta_y_rad / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)


def _validate_total_power(magnet_set: MagnetSet,
                         theta_x_windows: List[Tuple[float, float]],
                         theta_y_windows: List[Tuple[float, float]],
                         E_eV: np.ndarray,
                         E_J: np.ndarray,
                         T_E: np.ndarray,
                         settings: Settings) -> Optional[float]:
    if XrtBendingMagnet is None:
        return None
    # Build theta and psi sample grids across allowed windows
    def _concat_samples(windows: List[Tuple[float, float]], n: int) -> np.ndarray:
        parts = []
        for (a, b) in windows:
            if a == b:
                continue
            parts.append(np.linspace(a, b, max(2, n)))
        if not parts:
            return np.array([])
        return np.concatenate(parts)
    theta_s = _concat_samples(theta_x_windows, settings.validate_theta_samples)
    psi_s = _concat_samples(theta_y_windows, settings.validate_psi_samples)
    if theta_s.size == 0 or psi_s.size == 0:
        return 0.0
    # Require at least 2 samples for trapz integration
    if theta_s.size < 2 or psi_s.size < 2:
        return 0.0
    total_W = 0.0
    for mag in magnet_set.magnets:
        bm = XrtBendingMagnet()
        try: bm.eE = settings.energy_GeV
        except Exception: pass
        try: bm.B0 = mag.B_T
        except Exception: pass
        for _attr in ("I", "eI", "ringCurrent", "current"):
            if hasattr(bm, _attr):
                try:
                    setattr(bm, _attr, settings.current_A)
                    break
                except Exception:
                    continue
        # Compute Stokes over (E, theta, psi)
        stokes = bm.intensities_on_mesh(energy=E_eV, theta=theta_s, psi=psi_s)
        s0 = np.asarray(stokes[0])  # shape (nE, nTheta, nPsi[, harmonic])
        if s0.ndim == 4:
            s0 = s0.sum(axis=3)
        # Convert xrt S0 from per 0.1% bandwidth to per eV if requested
        if settings.s0_is_per_0p1bw:
            s0 = s0 / ((0.001 * E_eV)[:, None, None])
        # Apply attenuation and energy conversion
        s0E = s0 * (E_J * T_E)[:, None, None]
        # Integrate over energy -> I(theta, psi)
        I_th_psi = np.trapz(s0E, E_eV, axis=0)  # (nTheta, nPsi)
        # Integrate over psi -> I(theta)
        I_th = np.trapz(I_th_psi, psi_s, axis=1)  # (nTheta,)
        # Integrate over theta -> total_W
        total_W += float(np.trapz(I_th, theta_s))
    return total_W

def compute_screen_power(magnet_set: MagnetSet,
                         masks: List[Mask],
                         attenuators: List[Attenuator],
                         settings: Settings) -> ScreenResult:
    """
    Compute the total power density map at the screen for the selected magnet set.
    Steps:
      - Build energy grid
      - Keep only upstream masks and attenuators
      - Combine angular windows from masks; if none, X: union of magnet ranges; Y: ±N*sigma
      - Auto FOV at screen from angular windows
      - Build theta grids and allowed mask
      - For each magnet: get horizontal spectral-angle density via xrt BM model (or surrogate)
      - Build 2D via vertical Gaussian profile; apply mask; apply attenuation; integrate over energy
    """
    E_eV = _energy_grid(settings)
    # Convert photon energy to Joules for absolute power integration
    E_J = E_eV * 1.602176634e-19

    # Upstream filtering
    # Sort upstream masks by Z to ensure consistent behavior
    masks_up = sorted([m for m in masks if m.z_mm < settings.screen_z_mm], key=lambda m: m.z_mm)
    # Sort upstream attenuators by Z
    atts_up = sorted(filter_upstream(attenuators, settings.screen_z_mm), key=lambda a: a.z_mm)

    # Angular windows from masks
    theta_x_win_masks, theta_y_win_masks = combined_angular_windows_upstream(masks_up, settings.screen_z_mm)

    # Horizontal windows also constrained by magnet theta ranges
    theta_x_union_magnets = _horizontal_theta_union_from_magnets(magnet_set.magnets)
    # Intersect mask X windows with magnet X windows (simple intersection of unions)
    def intersect_unions(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not a:
            return b
        if not b:
            return a
        ia = sorted(a)
        ib = sorted(b)
        i, j = 0, 0
        out: List[Tuple[float, float]] = []
        while i < len(ia) and j < len(ib):
            s = max(ia[i][0], ib[j][0])
            e = min(ia[i][1], ib[j][1])
            if s <= e:
                out.append((s, e))
            if ia[i][1] < ib[j][1]:
                i += 1
            else:
                j += 1
        return out

    theta_x_windows = intersect_unions(theta_x_win_masks, theta_x_union_magnets) if theta_x_win_masks and theta_x_union_magnets else (theta_x_win_masks or theta_x_union_magnets)

    # Vertical windows: if masks define finite windows, use them; otherwise ±N*sigma
    def _finite_windows(ws: List[Tuple[float, float]]):
        return ws and all(math.isfinite(a) and math.isfinite(b) for (a, b) in ws)

    if _finite_windows(theta_y_win_masks):
        theta_y_windows = theta_y_win_masks
    else:
        sigma = _sigma_vert_rad(settings.energy_GeV)
        N = settings.vert_sigma_multiplier
        theta_y_windows = [(-N * sigma, +N * sigma)]

    # If either axis is fully blocked, return an empty but consistent result
    if not theta_x_windows or not theta_y_windows:
        x = np.linspace(-1, 1, settings.nx)
        y = np.linspace(-1, 1, settings.ny)
        power = np.zeros((settings.ny, settings.nx), dtype=float)
        line_power_x = np.zeros(settings.nx, dtype=float)
        line_power_y = np.zeros(settings.ny, dtype=float)
        z = settings.screen_z_mm
        theta_x_mrad = (x / z) * 1e3 if z != 0.0 else np.zeros_like(x)
        line_power_theta = np.zeros(settings.nx, dtype=float)
        # Transmission defined even in blocked case
        T_E = np.array(transmission_curve(atts_up, E_eV)) if atts_up else np.ones_like(E_eV)
        return ScreenResult(
            x_mm=x,
            y_mm=y,
            power_W_m2=power,
            line_power_x_W_per_mm=line_power_x,
            line_power_y_W_per_mm=line_power_y,
            theta_x_mrad=theta_x_mrad,
            line_power_theta_W_per_mrad=line_power_theta,
            total_power_W=0.0,
            energies_eV=E_eV,
            transmission_E=T_E,
            theta_x_windows_rad=theta_x_windows or [],
            theta_y_windows_rad=theta_y_windows or [],
            validation_total_W=None,
            validation_delta_percent=None,
            unmasked_total_W=None,
            allowed_fraction_percent=0.0,
            allowed_count=0,
            angular_total_W=0.0,
            delta_screen_vs_angular_percent=None,
            delta_angular_vs_validation_percent=None,
        )

    # Auto FOV
    x_mm, y_mm = _auto_fov_mm(theta_x_windows, theta_y_windows, settings)
    theta_x_rad, theta_y_rad = _theta_grids_from_screen(x_mm, y_mm, settings.screen_z_mm)

    # Allowed angular mask
    allowed_mask = _apply_angular_windows_mask(theta_x_rad, theta_y_rad, theta_x_windows, theta_y_windows)
    # Diagnostics: fraction of grid allowed by masks
    allowed_count = int(np.count_nonzero(allowed_mask))
    total_pixels = int(allowed_mask.size)
    allowed_fraction_percent = (100.0 * allowed_count / total_pixels) if total_pixels > 0 else 0.0

    # --- Begin: User-specified calculation (ignore xrt; use line power density formula) ---
    # Pixel sizes
    dx_mm = (x_mm[1] - x_mm[0]) if x_mm.size > 1 else 0.0
    dy_mm = (y_mm[1] - y_mm[0]) if y_mm.size > 1 else 0.0

    # Initialize power density (W/mm^2) and totals
    power_density = np.zeros((y_mm.size, x_mm.size), dtype=float)
    total_power_from_theta = 0.0

    # Precompute horizontal angles per column (rad) and active allowed columns
    theta_x_rad_grid = theta_x_rad
    allowed_cols = (allowed_mask.sum(axis=0) > 0)

    # Helper: intersect windows with magnet range and compute total width (rad)
    def _intersect_with_magnet(theta_windows: List[Tuple[float, float]], th_min: float, th_max: float) -> List[Tuple[float, float]]:
        if not theta_windows:
            return []
        i_min, i_max = (min(th_min, th_max), max(th_min, th_max))
        out = []
        for a, b in theta_windows:
            s = max(a, i_min)
            e = min(b, i_max)
            if s <= e:
                out.append((s, e))
        return out

    for mag in magnet_set.magnets:
        # Line power density per mrad (W/mrad)
        L_theta = 4.22 * mag.B_T * (settings.energy_GeV ** 3) * settings.current_A
        # Distance from magnet to screen (mm); skip invalid/zero distances
        z_mag_mm = settings.screen_z_mm - mag.z_mm
        if z_mag_mm <= 0:
            continue
        # Convert to line power density per mm: L_x = L_theta * 1000 / z
        L_x = L_theta * 1e3 / z_mag_mm

        # Magnet horizontal range in rad
        th_min = mrad_to_rad(mag.theta_min_mrad)
        th_max = mrad_to_rad(mag.theta_max_mrad)

        # Allowed theta windows (from masks) intersected with this magnet's range
        masked_theta = _intersect_with_magnet(theta_x_windows, th_min, th_max)
        # Total allowed theta width (in mrad)
        width_mrad = 0.0
        for s_th, e_th in masked_theta:
            width_mrad += (e_th - s_th) * 1e3
        # Magnet total power via theta-width
        P_magnet = L_theta * width_mrad
        total_power_from_theta += P_magnet

        # Distribute into the 2D map: for each allowed x-column within magnet theta range
        active_x = (theta_x_rad_grid >= min(th_min, th_max)) & (theta_x_rad_grid <= max(th_min, th_max))
        for j in np.where(active_x & allowed_cols)[0]:
            # Allowed vertical length at this x (mm)
            y_allow_len = float(np.count_nonzero(allowed_mask[:, j])) * dy_mm
            if y_allow_len <= 0:
                continue
            # Per-pixel density so that column integral over y equals L_x (W/mm)
            # Column power over width dx: L_x * dx
            # Area integral over column: sum(density * dy) * dx = L_x * dx -> density = L_x / y_allow_len
            density_col = L_x / y_allow_len
            col_mask = allowed_mask[:, j]
            power_density[col_mask, j] += density_col

    # Total power from area integral (W)
    total_power = float(np.sum(power_density) * (dx_mm * dy_mm))

    # Line power densities (W/mm)
    line_power_x_W_per_mm = np.sum(power_density, axis=0) * dy_mm
    line_power_y_W_per_mm = np.sum(power_density, axis=1) * dx_mm

    # Line power density per horizontal mrad
    theta_x_mrad = theta_x_rad * 1e3
    line_power_theta_W_per_mrad = line_power_x_W_per_mm * (settings.screen_z_mm / 1e3)

    # Angular-grid diagnostic total equals sum from theta integration (no xrt now)
    angular_total = total_power_from_theta
    validation_total = None
    validation_delta = None
    unmasked_total = None

    # Build result and return early
    return ScreenResult(
        x_mm, y_mm, power_density,
        line_power_x_W_per_mm=line_power_x_W_per_mm,
        line_power_y_W_per_mm=line_power_y_W_per_mm,
        theta_x_mrad=theta_x_mrad,
        line_power_theta_W_per_mrad=line_power_theta_W_per_mrad,
        total_power_W=total_power,
        energies_eV=np.array([]),
        transmission_E=np.array([]),
        theta_x_windows_rad=theta_x_windows,
        theta_y_windows_rad=theta_y_windows,
        validation_total_W=validation_total,
        validation_delta_percent=validation_delta,
        unmasked_total_W=unmasked_total,
        allowed_fraction_percent=allowed_fraction_percent,
        allowed_count=allowed_count,
        angular_total_W=angular_total,
        delta_screen_vs_angular_percent=(100.0 * (total_power - angular_total) / angular_total) if angular_total else None,
        delta_angular_vs_validation_percent=None,
    )
    # --- End: User-specified calculation ---

    # Transmission from attenuators (energy dependent)
    T_E = np.array(transmission_curve(atts_up, E_eV)) if atts_up else np.ones_like(E_eV)

    # Vertical profile (energy-independent sigma per your formula)
    vy = _vertical_profile(theta_y_rad, settings)  # shape (ny,)

    # Initialize power density (W per mm^2 on screen). We'll convert from angular density via Jacobian.
    power_density = np.zeros((y_mm.size, x_mm.size), dtype=float)
    # Angular-grid accumulator I(θ,ψ) [W/sr] for diagnostic integration
    I_ang = np.zeros((y_mm.size, x_mm.size), dtype=float)

    # Jacobian from angles to coordinates at distance z: dΩ ≈ dtheta_x * dtheta_y, and area element on screen dA = dz^2 dΩ for small angles.
    # Power density at screen ≈ integral_E [ I(E, theta_x, theta_y) * T(E) ] / (z^2) dE

    z_mm = settings.screen_z_mm

    # Build theta_x grid spacing
    # Use nonuniform grid in x mapped from screen x; compute local dtheta_x and dtheta_y from diffs
    dtheta_x = np.gradient(theta_x_rad)
    dtheta_y = np.gradient(theta_y_rad)

    # For each magnet, compute horizontal spectral density and accumulate
    for mag in magnet_set.magnets:
        # Limit horizontal theta to magnet's own range via mask to save compute
        th_min = mrad_to_rad(mag.theta_min_mrad)
        th_max = mrad_to_rad(mag.theta_max_mrad)
        active_x = (theta_x_rad >= min(th_min, th_max)) & (theta_x_rad <= max(th_min, th_max))
        if not np.any(active_x):
            continue

        spectral_h = _bm_horizontal_spectral_angle_density(mag, E_eV, theta_x_rad[active_x], settings)  # (nE, nTxActive)

        # Build 2D separable model: I(E, x, y) ≈ spectral_h(E, x) * vy(y)
        # Convert xrt S0 from per 0.1% bandwidth to per eV if requested
        if settings.s0_is_per_0p1bw:
            spectral_h = spectral_h / ((0.001 * E_eV)[:, None])
        # Apply energy-dependent attenuation and convert to power using photon energy (J)
        spectral_h *= (T_E * E_J)[:, None]

        # Integrate over energy: sum over E with weights dE
        if settings.logE:
            # Compute energy bin widths in linear space for integration
            dE = np.gradient(E_eV)
        else:
            dE = np.gradient(E_eV)

        I_x = np.trapz(spectral_h, E_eV, axis=0)  # shape (nTxActive,)

        # Expand to 2D via outer product with vertical profile (vy integrates to 1 over ψ)
        I_xy = np.outer(vy, I_x)  # shape (ny, nTxActive), units W/sr

        # Accumulate into angular-grid diagnostic (I(θ,ψ)) before any mapping
        I_ang[:, active_x] += I_xy

        # Map into full grid and convert to power density via far-field relation: power_density = I / z^2 (W/mm^2)
        tmp = np.zeros_like(power_density)
        tmp[:, active_x] = I_xy / (z_mm ** 2)
        power_density += tmp

    # Apply allowed angular mask (anything outside windows set to zero)
    allowed_float = allowed_mask.astype(float)
    power_density *= allowed_float

    # Angular-grid diagnostic: apply mask to I(θ,ψ) and integrate over θ and ψ using trapz
    I_ang *= allowed_float  # mask angular domain consistently
    try:
        I_theta = np.trapz(I_ang, theta_y_rad, axis=0)  # integrate over ψ -> function of θ
        angular_total = float(np.trapz(I_theta, theta_x_rad)) if I_theta.size > 1 else 0.0
    except Exception:
        angular_total = None

    # Line power densities (W/mm)
    dx_mm = (x_mm[1] - x_mm[0]) if x_mm.size > 1 else 0.0
    dy_mm = (y_mm[1] - y_mm[0]) if y_mm.size > 1 else 0.0
    line_power_x_W_per_mm = np.sum(power_density, axis=0) * dy_mm  # integrate over y
    line_power_y_W_per_mm = np.sum(power_density, axis=1) * dx_mm  # integrate over x

    # Line power density per horizontal mrad: Lθ(θ) = Lx(x) * z ; convert to per mrad by dividing by 1e3
    theta_x_mrad = theta_x_rad * 1e3
    line_power_theta_W_per_mrad = line_power_x_W_per_mm * (z_mm / 1e3)

    # Total power from area integral (W/mm^2 * mm^2)
    total_power = float(np.sum(power_density) * (dx_mm * dy_mm))

    # Optional validation using xrt integration across theta and psi
    validation_total = None
    validation_delta = None
    unmasked_total = None
    if settings.do_validation and XrtBendingMagnet is not None:
        try:
            # Masked validation (using current windows)
            if theta_x_windows and theta_y_windows:
                validation_total = _validate_total_power(magnet_set, theta_x_windows, theta_y_windows, E_eV, E_J, T_E, settings)
                if validation_total is not None and validation_total > 0:
                    validation_delta = 100.0 * (total_power - validation_total) / validation_total
            # Unmasked validation (ignore masks; use magnet union and ±N*sigma vertically)
            theta_x_union_magnets = _horizontal_theta_union_from_magnets(magnet_set.magnets)
            sigma = _sigma_vert_rad(settings.energy_GeV)
            N = settings.vert_sigma_multiplier
            theta_y_full = [(-N * sigma, +N * sigma)]
            if theta_x_union_magnets and theta_y_full:
                unmasked_total = _validate_total_power(magnet_set, theta_x_union_magnets, theta_y_full, E_eV, E_J, T_E, settings)
        except Exception:
            pass

    # Angular vs validation deltas
    delta_screen_vs_angular = None
    delta_angular_vs_validation = None
    if angular_total is not None and angular_total > 0:
        delta_screen_vs_angular = 100.0 * (total_power - angular_total) / angular_total
    if angular_total is not None and validation_total is not None and validation_total > 0:
        delta_angular_vs_validation = 100.0 * (angular_total - validation_total) / validation_total

    return ScreenResult(
        x_mm, y_mm, power_density,
        line_power_x_W_per_mm=line_power_x_W_per_mm,
        line_power_y_W_per_mm=line_power_y_W_per_mm,
        theta_x_mrad=theta_x_mrad,
        line_power_theta_W_per_mrad=line_power_theta_W_per_mrad,
        total_power_W=total_power,
        energies_eV=E_eV,
        transmission_E=T_E,
        theta_x_windows_rad=theta_x_windows,
        theta_y_windows_rad=theta_y_windows,
        validation_total_W=validation_total,
        validation_delta_percent=validation_delta,
        unmasked_total_W=unmasked_total,
        allowed_fraction_percent=allowed_fraction_percent,
        allowed_count=allowed_count,
        angular_total_W=angular_total,
        delta_screen_vs_angular_percent=delta_screen_vs_angular,
        delta_angular_vs_validation_percent=delta_angular_vs_validation,
    )
