from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import math


@dataclass
class Aperture:
    """
    Rectangular aperture in a mask.

    Units:
      - width_mm, height_mm: full sizes in millimeters
      - center_x_mm, center_y_mm: offsets relative to beam axis in millimeters
    """
    width_mm: float
    height_mm: float
    center_x_mm: float
    center_y_mm: float


@dataclass
class Mask:
    """
    Mask located at a position along the beamline with one or more rectangular apertures.

    Units:
      - z_mm: millimeters (position along beamline; screen only considers masks with z_mm < screen_z_mm)
    """
    name: str
    z_mm: float
    apertures: List[Aperture] = field(default_factory=list)


def _interval_union(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Union of 1D intervals [a, b] with a <= b. Returns a list of non-overlapping, sorted intervals.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged


def _interval_intersection(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Intersection of two sets of intervals (each list is assumed to be a union of disjoint intervals).
    """
    result: List[Tuple[float, float]] = []
    i, j = 0, 0
    a_sorted = sorted(a, key=lambda x: x[0])
    b_sorted = sorted(b, key=lambda x: x[0])
    while i < len(a_sorted) and j < len(b_sorted):
        a_start, a_end = a_sorted[i]
        b_start, b_end = b_sorted[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start <= end:
            result.append((start, end))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result


def mask_angular_windows(mask: Mask) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Compute the horizontal and vertical angular acceptance windows for a single mask,
    using exact angle mapping: theta_x = arctan(x_mm / z_mm), theta_y = arctan(y_mm / z_mm) (radians).

    Returns:
      (theta_x_windows_rad, theta_y_windows_rad) where each is a list of [min, max] intervals in radians.
      Windows are the union across all apertures in the mask.
    """
    z = mask.z_mm
    if z == 0.0:
        # A mask at the source plane can't be projected; return empty windows.
        return [], []

    theta_x_intervals: List[Tuple[float, float]] = []
    theta_y_intervals: List[Tuple[float, float]] = []
    for ap in mask.apertures:
        half_w = ap.width_mm / 2.0
        half_h = ap.height_mm / 2.0
        x_min = ap.center_x_mm - half_w
        x_max = ap.center_x_mm + half_w
        y_min = ap.center_y_mm - half_h
        y_max = ap.center_y_mm + half_h
        # Convert mm at plane to angles relative to source using exact arctangent mapping.
        theta_x_min = math.atan(x_min / z)
        theta_x_max = math.atan(x_max / z)
        theta_y_min = math.atan(y_min / z)
        theta_y_max = math.atan(y_max / z)
        # Ensure proper ordering
        theta_x_intervals.append((min(theta_x_min, theta_x_max), max(theta_x_min, theta_x_max)))
        theta_y_intervals.append((min(theta_y_min, theta_y_max), max(theta_y_min, theta_y_max)))

    return _interval_union(theta_x_intervals), _interval_union(theta_y_intervals)


def combined_angular_windows_upstream(masks: List[Mask], screen_z_mm: float) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Combine angular acceptance of all masks upstream of the screen (z < screen_z_mm).
    For each axis:
      - Compute union across apertures for each mask (per-axis).
      - Intersect across masks in the beamline (only upstream masks considered).

    Returns:
      (theta_x_windows_rad, theta_y_windows_rad) as lists of intervals in radians.
      If no upstream masks, returns [(-inf, +inf)] for both axes.
    """
    # Sort upstream masks by Z to ensure consistent intersection order
    upstream_masks = sorted([m for m in masks if m.z_mm < screen_z_mm], key=lambda m: m.z_mm)
    if not upstream_masks:
        return [(-float('inf'), float('inf'))], [(-float('inf'), float('inf'))]

    # Start with the first (nearest downstream) mask's windows
    first_tx, first_ty = mask_angular_windows(upstream_masks[0])
    if not first_tx:
        tx = []
    else:
        tx = first_tx
    if not first_ty:
        ty = []
    else:
        ty = first_ty

    for mask in upstream_masks[1:]:
        mx, my = mask_angular_windows(mask)
        tx = _interval_intersection(tx, mx) if tx and mx else []
        ty = _interval_intersection(ty, my) if ty and my else []

        # Early exit if either axis is fully blocked
        if not tx or not ty:
            break

    # If any axis ended up empty, it's fully blocked
    if not tx:
        tx = []
    if not ty:
        ty = []

    return tx, ty