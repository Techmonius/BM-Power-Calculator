from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from scipy import ndimage
except Exception:
    ndimage = None

from .compute import ScreenResult


@dataclass
class Footprint:
    label: int
    # Sizes in mm
    width_x_mm: float
    height_y_mm: float
    # Sizes in mrad (theta and psi)
    width_theta_mrad: float
    height_psi_mrad: float
    # Bounding box indices (y, x)
    y_min_idx: int
    y_max_idx: int
    x_min_idx: int
    x_max_idx: int
    # Total power in Watts for this footprint
    total_power_W: float
    # Optional centroid in mm
    centroid_x_mm: float
    centroid_y_mm: float


def extract_footprints(
    res: ScreenResult,
    screen_z_mm: float,
    rel_threshold: float = 0.05,
    min_pixels: int = 20,
    connectivity: int = 1,
) -> List[Footprint]:
    """
    Extract contiguous beam footprints from the power density map.

    Args:
        res: ScreenResult from compute_screen_power
        screen_z_mm: Screen Z distance in mm (for converting mm -> mrad)
        rel_threshold: relative threshold vs. max power density (e.g., 0.05 for 5%)
        min_pixels: minimum number of pixels to keep a footprint
        connectivity: 1 (4-neighborhood) or 2 (8-neighborhood) for component labeling

    Returns:
        List of Footprint objects with sizes in mm and mrad and total power per footprint.
    """
    if res.power_W_m2.size == 0:
        return []

    data = np.array(res.power_W_m2, dtype=float)
    max_val = float(np.max(data))
    if max_val <= 0:
        return []

    thresh = max_val * rel_threshold
    mask = data > thresh

    # Label connected components
    if ndimage is None:
        # Simple fallback: no SciPy available. Use a naive BFS labeling (slower on large grids).
        labels = _label_components_naive(mask, connectivity=connectivity)
        n_labels = int(labels.max())
    else:
        structure = None
        if connectivity == 1:
            structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
        else:
            structure = np.ones((3,3), dtype=int)
        labels, n_labels = ndimage.label(mask, structure=structure)

    if n_labels == 0:
        return []

    # Pixel sizes
    dx_mm = float(res.x_mm[1] - res.x_mm[0]) if res.x_mm.size > 1 else 1.0
    dy_mm = float(res.y_mm[1] - res.y_mm[0]) if res.y_mm.size > 1 else 1.0
    pixel_area_mm2 = dx_mm * dy_mm

    footprints: List[Footprint] = []

    for lbl in range(1, n_labels + 1):
        indices = np.argwhere(labels == lbl)
        if indices.shape[0] < min_pixels:
            continue
        ys = indices[:, 0]
        xs = indices[:, 1]
        y_min = int(ys.min())
        y_max = int(ys.max())
        x_min = int(xs.min())
        x_max = int(xs.max())

        # Bounding box extents in mm (inclusive indices)
        width_x_mm = float(res.x_mm[x_max] - res.x_mm[x_min])
        height_y_mm = float(res.y_mm[y_max] - res.y_mm[y_min])

        # Convert to mrad via exact mapping at bounding edges
        x0 = float(res.x_mm[x_min])
        x1 = float(res.x_mm[x_max])
        y0 = float(res.y_mm[y_min])
        y1 = float(res.y_mm[y_max])
        width_theta_mrad = (np.arctan(x1 / screen_z_mm) - np.arctan(x0 / screen_z_mm)) * 1e3
        height_psi_mrad = (np.arctan(y1 / screen_z_mm) - np.arctan(y0 / screen_z_mm)) * 1e3

        # Total power in this footprint: integrate density over its pixels (W/mm^2 * mm^2)
        local_power_density = data[labels == lbl]
        total_power_W = float(np.sum(local_power_density) * pixel_area_mm2)

        # Power-weighted centroid (mm)
        powers = data[ys, xs]
        sumP = float(np.sum(powers))
        if sumP > 0.0:
            centroid_x_mm = float(np.sum(powers * res.x_mm[xs]) / sumP)
            centroid_y_mm = float(np.sum(powers * res.y_mm[ys]) / sumP)
        else:
            # Fallback to unweighted center if zero power (degenerate)
            cx_idx = int(np.round(xs.mean()))
            cy_idx = int(np.round(ys.mean()))
            centroid_x_mm = float(res.x_mm[cx_idx])
            centroid_y_mm = float(res.y_mm[cy_idx])

        footprints.append(
            Footprint(
                label=lbl,
                width_x_mm=width_x_mm,
                height_y_mm=height_y_mm,
                width_theta_mrad=width_theta_mrad,
                height_psi_mrad=height_psi_mrad,
                y_min_idx=y_min,
                y_max_idx=y_max,
                x_min_idx=x_min,
                x_max_idx=x_max,
                total_power_W=total_power_W,
                centroid_x_mm=centroid_x_mm,
                centroid_y_mm=centroid_y_mm,
            )
        )

    return footprints


def _label_components_naive(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """
    Very simple connected-component labeling as a fallback if SciPy is unavailable.
    Returns a labels array of same shape as mask.
    """
    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    lbl = 0

    # Neighbor offsets
    if connectivity == 1:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        neighbors = [
            (-1,-1),(-1,0),(-1,1),
            (0,-1),         (0,1),
            (1,-1), (1,0),  (1,1)
        ]

    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                lbl += 1
                # BFS/DFS
                stack = [(y, x)]
                labels[y, x] = lbl
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = lbl
                                stack.append((ny, nx))
    return labels


def footprints_to_table(footprints: List[Footprint]) -> List[Tuple]:
    """
    Convert footprints to a simple table (list of tuples) for easy printing/CSV.
    Columns: (label, width_x_mm, height_y_mm, width_theta_mrad, height_psi_mrad, total_power_W, centroid_x_mm, centroid_y_mm)
    """
    table = []
    for fp in footprints:
        table.append(
            (
                fp.label,
                fp.width_x_mm,
                fp.height_y_mm,
                fp.width_theta_mrad,
                fp.height_psi_mrad,
                fp.total_power_W,
                fp.centroid_x_mm,
                fp.centroid_y_mm,
            )
        )
    return table


def print_footprints_summary(footprints: List[Footprint]) -> None:
    if not footprints:
        print("No footprints detected (threshold too high or no signal).")
        return
    print("Detected footprints:")
    for fp in footprints:
        print(
            f" - Label {fp.label}: size = ({fp.width_x_mm:.3f} mm, {fp.height_y_mm:.3f} mm)"
            f" | angles = ({fp.width_theta_mrad:.3f} mrad, {fp.height_psi_mrad:.3f} mrad)"
            f" | total power = {fp.total_power_W:.3e} W"
            f" | centroid = ({fp.centroid_x_mm:.3f} mm, {fp.centroid_y_mm:.3f} mm)"
        )


def overlay_footprints_on_ax(ax, res: ScreenResult, footprints: List[Footprint]) -> None:
    """
    Overlay bounding boxes of footprints on an existing imshow axes.
    """
    import matplotlib.patches as patches
    for fp in footprints:
        x0 = res.x_mm[fp.x_min_idx]
        x1 = res.x_mm[fp.x_max_idx]
        y0 = res.y_mm[fp.y_min_idx]
        y1 = res.y_mm[fp.y_max_idx]
        rect = patches.Rectangle(
            (x0, y0), (x1 - x0), (y1 - y0),
            linewidth=1.5, edgecolor='magenta', facecolor='none', zorder=5
        )
        ax.add_patch(rect)
        ax.text(x0, y1, f"FP {fp.label}", color='magenta', fontsize=8, va='bottom', ha='left')
