from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from .magnets import Magnet, MagnetSet, get_available_magnet_sets, mrad_to_rad
from .masks import Mask, combined_angular_windows_upstream
from .attenuation import Attenuator, filter_upstream, transmission_curve

# xrt imports
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

    x_min_mm = math.tan(tx_min) * z
    x_max_mm = math.tan(tx_max) * z
    y_min_mm = math.tan(ty_min) * z
    y_max_mm = math.tan(ty_max) * z

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
    # Exact angle mapping: theta = arctan(coord / z)
    theta_x = np.arctan(x_mm / z_mm)
    theta_y = np.arctan(y_mm / z_mm)
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
    Surrogate horizontal spectral-angular density along theta_x for each energy.
    Returns array shape (nE, nThetaX) in arbitrary units scaled by B·E^3·I.
    """
    nE = energies_eV.size
    nTx = theta_x_rad.size
    spectral = np.zeros((nE, nTx), dtype=float)
    gamma = _gamma(settings.energy_GeV)
    theta_c = 1.0 / gamma
    for iE in range(nE):
        spectral[iE, :] = 1.0 / (1.0 + (theta_x_rad / theta_c) ** 2) ** 2
    spectral *= magnet.B_T * settings.energy_GeV**3 * settings.current_A
    return spectral    # xrt BM expects: eE (electron energy in GeV), I (A), B0 (T)
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


def _validate_total_power(*args, **kwargs) -> Optional[float]:
    # Validation disabled (xrt removed)
    return None

def compute_screen_power(magnet_set: MagnetSet,
                         masks: List[Mask],
                         attenuators: List[Attenuator],
                         settings: Settings) -> ScreenResult:
    """
    Simplified core calculation (no spectral/xrt/attenuators):
      - Exact angle mapping: x = z*tan(theta), y = z*tan(psi)
      - Vertical distribution: normalized Gaussian vy(psi) with sigma = 0.608/gamma
      - Horizontal line power per mrad: Lθ = 4.22·B·E^3·I (constant within magnet acceptance)
      - Exact Jacobian mapping to screen PD: PD = I / (z^2 * (1+(x/z)^2) * (1+(y/z)^2))
      - Half-open magnet intervals; most downstream magnet owns the shared upper boundary
    """
    # Upstream masks only
    masks_up = sorted([m for m in masks if m.z_mm < settings.screen_z_mm], key=lambda m: m.z_mm)

    # Angular windows from masks
    theta_x_win_masks, theta_y_win_masks = combined_angular_windows_upstream(masks_up, settings.screen_z_mm)

    # Constrain horizontal by magnet union
    theta_x_union_magnets = _horizontal_theta_union_from_magnets(magnet_set.magnets)

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

    # Vertical windows: use masks if finite; otherwise ±N*sigma
    def _finite_windows(ws: List[Tuple[float, float]]):
        return ws and all(math.isfinite(a) and math.isfinite(b) for (a, b) in ws)

    if _finite_windows(theta_y_win_masks):
        theta_y_windows = theta_y_win_masks
    else:
        sigma = _sigma_vert_rad(settings.energy_GeV)
        N = settings.vert_sigma_multiplier
        theta_y_windows = [(-N * sigma, +N * sigma)]

    # If blocked, return empty result with consistent shapes
    if not theta_x_windows or not theta_y_windows:
        x = np.linspace(-1, 1, settings.nx)
        y = np.linspace(-1, 1, settings.ny)
        power = np.zeros((settings.ny, settings.nx), dtype=float)
        line_power_x = np.zeros(settings.nx, dtype=float)
        line_power_y = np.zeros(settings.ny, dtype=float)
        z = settings.screen_z_mm
        theta_x_mrad = np.arctan(x / z) * 1e3 if z != 0.0 else np.zeros_like(x)
        line_power_theta = np.zeros(settings.nx, dtype=float)
        return ScreenResult(
            x_mm=x,
            y_mm=y,
            power_W_m2=power,
            line_power_x_W_per_mm=line_power_x,
            line_power_y_W_per_mm=line_power_y,
            theta_x_mrad=theta_x_mrad,
            line_power_theta_W_per_mrad=line_power_theta,
            total_power_W=0.0,
            energies_eV=np.array([]),
            transmission_E=np.array([]),
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

    # Auto FOV and angle grids
    x_mm, y_mm = _auto_fov_mm(theta_x_windows, theta_y_windows, settings)
    theta_x_rad, theta_y_rad = _theta_grids_from_screen(x_mm, y_mm, settings.screen_z_mm)

    # Allowed angular mask and diagnostics
    allowed_mask = _apply_angular_windows_mask(theta_x_rad, theta_y_rad, theta_x_windows, theta_y_windows)
    allowed_count = int(np.count_nonzero(allowed_mask))
    total_pixels = int(allowed_mask.size)
    allowed_fraction_percent = (100.0 * allowed_count / total_pixels) if total_pixels > 0 else 0.0

    # Vertical profile vy(psi) and init PD
    vy = _vertical_profile(theta_y_rad, settings)  # shape (ny,)
    power_density = np.zeros((y_mm.size, x_mm.size), dtype=float)

    # Exact mapping factors
    z_mm = settings.screen_z_mm
    fy = (1.0 + (y_mm / z_mm) ** 2)[:, None]  # (ny,1)

    # Most downstream magnet owns the upper boundary
    downstream_mag = max(magnet_set.magnets, key=lambda m: m.z_mm) if magnet_set.magnets else None

    # Angular total diagnostic [W]
    angular_total = 0.0

    # Helper: intersect windows with magnet acceptance using half-open intervals
    def _intersect_windows_half_open(windows: List[Tuple[float, float]], th_lo: float, th_hi: float, include_upper: bool):
        out: List[Tuple[float, float]] = []
        lo = min(th_lo, th_hi)
        hi = max(th_lo, th_hi)
        for a, b in (windows or []):
            s = max(a, lo)
            e = min(b, hi)
            if include_upper:
                if s <= e:
                    out.append((s, e))
            else:
                if s < e:
                    out.append((s, e))
        return out

    # Accumulate PD from magnets
    for mag in magnet_set.magnets:
        th_min = mrad_to_rad(mag.theta_min_mrad)
        th_max = mrad_to_rad(mag.theta_max_mrad)
        include_upper = (mag is downstream_mag)

        masked_theta = _intersect_windows_half_open(theta_x_windows, th_min, th_max, include_upper)

        if include_upper:
            active_x = (theta_x_rad >= min(th_min, th_max)) & (theta_x_rad <= max(th_min, th_max))
        else:
            active_x = (theta_x_rad >= min(th_min, th_max)) & (theta_x_rad < max(th_min, th_max))

        allowed_cols = (allowed_mask.sum(axis=0) > 0)
        active_cols = np.where(active_x & allowed_cols)[0]
        if active_cols.size == 0:
            continue

        # Line power per mrad (constant within acceptance)
        L_theta_mrad = 4.22 * mag.B_T * (settings.energy_GeV ** 3) * settings.current_A  # [W/mrad]
        L_theta_rad = L_theta_mrad * 1e3  # [W/rad]

        # Angular diagnostic total via Δθ (mrad)
        width_mrad = 0.0
        for s_th, e_th in masked_theta:
            width_mrad += (e_th - s_th) * 1e3
        angular_total += L_theta_mrad * width_mrad

        for j in active_cols:
            th_j = theta_x_rad[j]
            # Check if θ_j lies in any masked theta interval with half-open rule
            in_any = False
            for s_th, e_th in masked_theta:
                if include_upper:
                    if (th_j >= s_th) and (th_j <= e_th):
                        in_any = True
                        break
                else:
                    if (th_j >= s_th) and (th_j < e_th):
                        in_any = True
                        break
            if not in_any:
                continue

            fx_j = (1.0 + (x_mm[j] / z_mm) ** 2)
            # Angular intensity column [W/sr] at θ_j: separable vy(ψ) times Lθ (per-rad)
            I_col = vy * L_theta_rad  # (ny,)
            # Exact Jacobian mapping to PD and apply y-mask
            pd_col = (I_col / ((z_mm ** 2) * fx_j)) / fy[:, 0]
            col_mask = allowed_mask[:, j]
            power_density[col_mask, j] += pd_col[col_mask]

    # Apply mask (safety)
    power_density *= allowed_mask.astype(float)

    # Summaries
    dx_mm = (x_mm[1] - x_mm[0]) if x_mm.size > 1 else 0.0
    dy_mm = (y_mm[1] - y_mm[0]) if y_mm.size > 1 else 0.0
    line_power_x_W_per_mm = np.sum(power_density, axis=0) * dy_mm
    line_power_y_W_per_mm = np.sum(power_density, axis=1) * dx_mm

    # Line power per mrad with exact dx/dθ
    theta_x_mrad = theta_x_rad * 1e3
    jac_dx_dtheta = z_mm * (1.0 + (x_mm / z_mm) ** 2)
    line_power_theta_W_per_mrad = line_power_x_W_per_mm * (jac_dx_dtheta / 1e3)

    total_power = float(np.sum(power_density) * (dx_mm * dy_mm))
    validation_total = None
    validation_delta = None
    unmasked_total = None

    # Angular vs screen delta
    delta_screen_vs_angular = None
    if angular_total is not None and angular_total > 0:
        delta_screen_vs_angular = 100.0 * (total_power - angular_total) / angular_total

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
        delta_screen_vs_angular_percent=delta_screen_vs_angular,
        delta_angular_vs_validation_percent=None,
    )
