from __future__ import annotations

from typing import List
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import json
import os
try:
    import xraylib
except Exception:
    xraylib = None

from .magnets import get_available_magnet_sets
from .masks import Mask, Aperture
from .attenuation import Attenuator
from .compute import Settings, compute_screen_power
from .footprints import extract_footprints, overlay_footprints_on_ax
from .ui_presets import (
    default_settings,
    
    
    default_magnet_set_title,
    materials_dropdown_options,
)

# ---- Helper widget classes ----

class ApertureWidget:
    def __init__(self, idx: int, ap: Optional[Aperture] = None, on_remove=None):
        self.idx = idx
        ap = ap or Aperture(width_mm=1.0, height_mm=1.0, center_x_mm=0.0, center_y_mm=0.0)
        self._on_remove = on_remove
        self.btn_remove = widgets.Button(description="−", tooltip="Remove aperture", layout=widgets.Layout(width="40px"))
        self.btn_remove.on_click(lambda _: self._on_remove and self._on_remove(self))
        # Store explicit references to inputs to avoid relying on children order
        self.label = widgets.Label(value=f"Aperture {idx}")
        self.cx = widgets.FloatText(value=ap.center_x_mm, description="cx [mm]", layout=widgets.Layout(width="150px"))
        self.cy = widgets.FloatText(value=ap.center_y_mm, description="cy [mm]", layout=widgets.Layout(width="150px"))
        self.w_mm = widgets.FloatText(value=ap.width_mm, description="w [mm]", layout=widgets.Layout(width="150px"))
        self.h_mm = widgets.FloatText(value=ap.height_mm, description="h [mm]", layout=widgets.Layout(width="150px"))
        self.w = widgets.HBox([
            self.label,
            self.cx,
            self.cy,
            self.w_mm,
            self.h_mm,
            self.btn_remove,
        ])

    def to_model(self) -> Aperture:
        return Aperture(width_mm=self.w_mm.value, height_mm=self.h_mm.value, center_x_mm=self.cx.value, center_y_mm=self.cy.value)

class MaskWidget:
    def __init__(self, idx: int, m: Optional[Mask] = None, on_remove=None):
        self.idx = idx
        m = m or Mask(name=f"Mask{idx}", z_mm=0.0, apertures=[])
        self.name = widgets.Text(value=m.name, description="Name", layout=widgets.Layout(width="220px"))
        self.z = widgets.FloatText(value=m.z_mm, description="Z [mm]", layout=widgets.Layout(width="180px"))
        self.apertures: List[ApertureWidget] = []
        for i, ap in enumerate(m.apertures):
            self.apertures.append(ApertureWidget(i + 1, ap, on_remove=self._on_remove_aperture))
        self.btn_add_ap = widgets.Button(description="+ Aperture", button_style='info')
        self.btn_add_ap.on_click(self._on_add_aperture)
        self._on_remove = on_remove
        self.btn_remove_mask = widgets.Button(description="− Mask", button_style='danger')
        self.btn_remove_mask.on_click(lambda _: self._on_remove and self._on_remove(self))
        self.ap_container = widgets.VBox([ap.w for ap in self.apertures])
        header = widgets.HBox([widgets.Label(value=f"Mask {idx}"), self.name, self.z, self.btn_add_ap, self.btn_remove_mask])
        self.w = widgets.VBox([header, self.ap_container])

    def _on_add_aperture(self, _):
        apw = ApertureWidget(len(self.apertures) + 1, on_remove=self._on_remove_aperture)
        self.apertures.append(apw)
        self.ap_container.children = [ap.w for ap in self.apertures]

    def _on_remove_aperture(self, apw):
        try:
            self.apertures.remove(apw)
            self.ap_container.children = [ap.w for ap in self.apertures]
        except ValueError:
            pass

    def to_model(self) -> Mask:
        return Mask(name=self.name.value, z_mm=self.z.value, apertures=[apw.to_model() for apw in self.apertures])

class AttenuatorWidget:
    def __init__(self, idx: int, a: Optional[Attenuator] = None, materials: Optional[List[str]] = None, on_remove=None):
        self.idx = idx
        materials = materials or materials_dropdown_options()
        a = a or Attenuator(z_mm=0.0, material=materials[0], thickness_mm=0.1)
        self.title = widgets.Label(value=f"Attenuator {idx}")
        self.z = widgets.FloatText(value=a.z_mm, description="Z [mm]", layout=widgets.Layout(width="180px"))
        self.material = widgets.Dropdown(options=materials, value=a.material, description="Material", layout=widgets.Layout(width="220px"))
        self.thickness = widgets.FloatText(value=a.thickness_mm, description="Thk [mm]", layout=widgets.Layout(width="180px"))
        self._on_remove = on_remove
        self.btn_remove_att = widgets.Button(description="− Attenuator", button_style='danger')
        self.btn_remove_att.on_click(lambda _: self._on_remove and self._on_remove(self))
        self.w = widgets.HBox([self.title, self.z, self.material, self.thickness, self.btn_remove_att])

    def to_model(self) -> Attenuator:
        return Attenuator(z_mm=self.z.value, material=self.material.value, thickness_mm=self.thickness.value)

# ---- Main UI ----

class BeamlineUI:
    def __init__(self):
        self.magnet_sets = get_available_magnet_sets()
        self.settings = default_settings()

        # Settings widgets
        self.dd_set = widgets.Dropdown(options=list(self.magnet_sets.keys()), value=default_magnet_set_title(), description="Magnet set")
        self.energy = widgets.FloatText(value=self.settings.energy_GeV, description="E [GeV]")
        self.current = widgets.FloatText(value=self.settings.current_A, description="I [A]")
        self.screen_z = widgets.FloatText(value=self.settings.screen_z_mm, description="Screen Z [mm]")
        self.nx = widgets.IntText(value=self.settings.nx, description="Nx")
        self.ny = widgets.IntText(value=self.settings.ny, description="Ny")
        self.Emin = widgets.FloatText(value=self.settings.Emin_eV, description="Emin [eV]")
        self.Emax = widgets.FloatText(value=self.settings.Emax_eV, description="Emax [eV]")
        self.nE = widgets.IntText(value=self.settings.nE, description="nE")
        self.logE = widgets.Checkbox(value=self.settings.logE, description="Log spacing")
        self.vert_sigma = widgets.FloatText(value=getattr(self.settings, 'vert_sigma_multiplier', 5.0), description="Vert Nsigma", layout=widgets.Layout(width="160px"))

        settings_row1 = widgets.HBox([self.dd_set, self.energy, self.current, self.screen_z])
        settings_row2 = widgets.HBox([self.nx, self.ny, self.Emin, self.Emax, self.nE, self.logE, self.vert_sigma])

        # Masks
        self.masks_widgets: List[MaskWidget] = []
        self.btn_add_mask = widgets.Button(description="+ Mask", button_style='info')
        self.btn_add_mask.on_click(self._on_add_mask)
        self.masks_container = widgets.VBox([mw.w for mw in self.masks_widgets])
        masks_box = widgets.VBox([widgets.HTML("<b>Masks</b>"), self.btn_add_mask, self.masks_container])

        # Attenuators
        self.att_widgets: List[AttenuatorWidget] = []
        self.btn_add_att = widgets.Button(description="+ Attenuator", button_style='info')
        self.btn_add_att.on_click(self._on_add_att)
        self.atts_container = widgets.VBox([aw.w for aw in self.att_widgets])
        atts_box = widgets.VBox([widgets.HTML("<b>Attenuators</b>"), self.btn_add_att, self.atts_container])

        # Run / output
        self.btn_run = widgets.Button(description="Compute", button_style='success')
        self.btn_run.on_click(self._on_run)
        self.output = widgets.Output()

        settings_box = widgets.VBox([settings_row1, settings_row2])
        self.accordion = widgets.Accordion(children=[settings_box, masks_box, atts_box])
        self.accordion.set_title(0, "Settings")
        self.accordion.set_title(1, "Masks")
        self.accordion.set_title(2, "Attenuators")
        self.units_note = widgets.HTML("<div style='color:#555;margin:6px 0'>Units: Screen power density W/mm^2; Angular power density W/mrad^2; Angles are computed from geometry.</div>")
        self.css = widgets.HTML("<style>.bm-table{border-collapse:collapse;font-size:12px;border:1px solid #ddd;}.bm-table th,.bm-table td{padding:6px 10px;border:1px solid #ddd;}.bm-table th{background:#f7f7f7;text-align:left;}.bm-table tr:nth-child(even){background:#fafafa;}.bm-num{text-align:right;}</style>")
        self.root = widgets.VBox([
            widgets.HTML("<h3>Bending Magnet Power at Screen</h3>"),
            self.css,
            self.accordion,
            self.units_note,
            self.btn_run,
            self.output,
        ])
        # Try loading last saved scenario
        self._load_state()

    def _on_add_mask(self, _):
        mw = MaskWidget(len(self.masks_widgets) + 1, on_remove=self._on_remove_mask)
        self.masks_widgets.append(mw)
        self.masks_container.children = [mw.w for mw in self.masks_widgets]

    def _on_add_att(self, _):
        aw = AttenuatorWidget(len(self.att_widgets) + 1, materials=materials, on_remove=self._on_remove_att)
        self.att_widgets.append(aw)
        self.atts_container.children = [aw.w for aw in self.att_widgets]

    def _collect_models(self):
        # Settings
        s = Settings(
            distances_unit="mm",
            energy_GeV=self.energy.value,
            current_A=self.current.value,
            screen_z_mm=self.screen_z.value,
            nx=self.nx.value,
            ny=self.ny.value,
            Emin_eV=self.Emin.value,
            Emax_eV=self.Emax.value,
            nE=self.nE.value,
            logE=self.logE.value,
            vert_sigma_multiplier=self.vert_sigma.value,
        )
        masks = [mw.to_model() for mw in self.masks_widgets]
        atts = [aw.to_model() for aw in self.att_widgets]
        mset = self.magnet_sets[self.dd_set.value]
        return s, masks, atts, mset

    def _on_remove_mask(self, mw: MaskWidget):
        try:
            self.masks_widgets.remove(mw)
            self.masks_container.children = [m.w for m in self.masks_widgets]
        except ValueError:
            pass

    def _on_remove_att(self, aw: AttenuatorWidget):
        try:
            self.att_widgets.remove(aw)
            self.atts_container.children = [a.w for a in self.att_widgets]
        except ValueError:
            pass

    def _on_run(self, _):
        with self.output:
            clear_output()
            css_html = """<style>
.bm-table{border-collapse:collapse;font-size:12px;border:1px solid #ddd;}
.bm-table th,.bm-table td{padding:6px 10px;border:1px solid #ddd;}
.bm-table th{background:#f7f7f7;text-align:left;}
.bm-table tr:nth-child(even){background:#fafafa;}
.bm-num{text-align:right;}
</style>"""
            display(widgets.HTML(css_html))
            # Warning banner if xraylib missing
            if xraylib is None:
                display(widgets.HTML("<div style='padding:8px;border:1px solid #cc7;background:#fff3cd;color:#856404;'><b>Warning:</b> Attenuation is disabled because xraylib is not installed. Results exclude attenuation.</div>"))
            print("Computing... (up to ~60s)")
            s, masks, atts, mset = self._collect_models()
            try:
                res = compute_screen_power(mset, masks, atts, s)
            except Exception as e:
                print("Error:", e)
                return

            # Plot power density
            figPD, axPD = plt.subplots(1, 2, figsize=(12, 5))
            # Build Y-profile and screen power density maps
            extent_mm = [res.x_mm[0], res.x_mm[-1], res.y_mm[0], res.y_mm[-1]]
                        # Build Y-profile (mean power density across x) and screen power density map
            pd_y_mean = np.mean(res.power_W_m2, axis=1)
            axPD[0].plot(pd_y_mean, res.y_mm)
            axPD[0].set_title("Power density vs y at screen")
            axPD[0].set_xlabel("Power density [W/mm^2]")
            axPD[0].set_ylabel("y [mm]")
            axPD[0].grid(True, ls=":")
            # Right: screen power density [W/mm^2]
            im_mm = axPD[1].imshow(res.power_W_m2, extent=extent_mm, origin="lower", aspect="equal")
            axPD[1].set_title("Power density at screen [W/mm^2]")
            axPD[1].set_xlabel("x [mm]")
            axPD[1].set_ylabel("y [mm]")
            axPD[1].invert_xaxis()
            plt.colorbar(im_mm, ax=axPD[1])
            

            # Extract beam footprints from the power density map (defaults: 5% threshold)
            fps = extract_footprints(res, screen_z_mm=s.screen_z_mm, rel_threshold=0.01, min_pixels=5, connectivity=1)
            try:
                overlay_footprints_on_ax(axPD[1], res, fps)
            except Exception:
                pass
            

            # Show the two PD plots
            plt.tight_layout()
            plt.show()

            # Transmission curve (shown only if there are upstream attenuators and data available)
            has_upstream_atts = any(a.z_mm < s.screen_z_mm for a in atts)
            if has_upstream_atts and getattr(res, 'energies_eV', None) is not None and np.size(res.energies_eV) > 0 and np.size(res.transmission_E) > 0:
                figT, axT = plt.subplots(1, 1, figsize=(6, 4))
                axT.plot(res.energies_eV, res.transmission_E)
                axT.set_xscale("log")
                axT.set_xlabel("Energy [eV]")
                axT.set_ylabel("Transmission")
                axT.set_title("Upstream attenuation")
                axT.grid(True, which="both", ls=":")
                plt.tight_layout()
                plt.show()

            # Footprints summary table
            if fps:
                def _fmt(v, nd=3):
                    try:
                        return f"{float(v):.{nd}f}"
                    except Exception:
                        return str(v)
                rows_html = []
                rows_html.append("<tr><th style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;text-align:left'>Footprint</th><th style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;text-align:left'>Size X [mm]</th><th style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;text-align:left'>Size Y [mm]</th><th style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;text-align:left'>Width theta [mrad]</th><th style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;text-align:left'>Height psi [mrad]</th><th style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;text-align:left'>Total Power [W]</th></tr>")
                total_fp_power = 0.0
                for fp in fps:
                    total_fp_power += float(fp.total_power_W)
                    rows_html.append(
                        f"<tr>"
                        f"<td style='padding:6px 10px;border:1px solid #ddd;text-align:left'>{fp.label}</td>"
                        f"<td style='padding:6px 10px;border:1px solid #ddd;text-align:right'>{_fmt(fp.width_x_mm)}</td>"
                        f"<td style='padding:6px 10px;border:1px solid #ddd;text-align:right'>{_fmt(fp.height_y_mm)}</td>"
                        f"<td style='padding:6px 10px;border:1px solid #ddd;text-align:right'>{_fmt(fp.width_theta_mrad)}</td>"
                        f"<td style='padding:6px 10px;border:1px solid #ddd;text-align:right'>{_fmt(fp.height_psi_mrad)}</td>"
                        f"<td style='padding:6px 10px;border:1px solid #ddd;text-align:right'>{_fmt(fp.total_power_W)}</td>"
                        f"</tr>"
                    )
                # Add a bottom sum row
                rows_html.append(
                    f"<tr style=\"border-top:1px solid #ccc\"><th colspan=5 style=\"text-align:right\">Sum of footprint powers [W]</th><th style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;text-align:left'>{_fmt(total_fp_power)}</th></tr>"
                )
                table_html = (
                    "<div style=\"margin-top:8px\">"
                    "<h4 style=\"margin:4px 0\">Footprints</h4>"
                    "<table style=\"border-collapse:collapse;font-size:12px\">" + "".join(rows_html) + "</table>" +
                    "</div>"
                )
                display(widgets.HTML(table_html))
            else:
                display(widgets.HTML("<div style=\"margin-top:8px;color:#666\">No footprints detected (threshold too high or no signal).</div>"))

            # ---- Ray-tracing style envelope views (top-down and side-profile) ----
            # Helpers for intervals
            def _merge_intervals(intervals):
                if not intervals:
                    return []
                intervals = sorted([(min(a,b), max(a,b)) for (a,b) in intervals], key=lambda x: x[0])
                merged = [intervals[0]]
                for s0,e0 in intervals[1:]:
                    s,e = merged[-1]
                    if s0 <= e:
                        merged[-1] = (s, max(e, e0))
                    else:
                        merged.append((s0, e0))
                return merged
            def _intersect_unions(a, b):
                if not a or not b:
                    return []
                a = sorted(a); b = sorted(b)
                i=j=0; out=[]
                while i<len(a) and j<len(b):
                    s=max(a[i][0], b[j][0]); e=min(a[i][1], b[j][1])
                    if s<=e:
                        out.append((s,e))
                    if a[i][1] < b[j][1]: i+=1
                    else: j+=1
                return out

            # Build initial angular windows
            # Horizontal: union of magnet theta ranges
            tx0 = []
            for mag in mset.magnets:
                th_min = min(mag.theta_min_mrad, mag.theta_max_mrad) * 1e-3
                th_max = max(mag.theta_min_mrad, mag.theta_max_mrad) * 1e-3
                tx0.append((th_min, th_max))
            tx_cur = _merge_intervals(tx0)
            # Vertical: ±N*sigma (unless constrained by masks later)
            sigma = 0.608/(1957.0*s.energy_GeV)
            N = getattr(s, 'vert_sigma_multiplier', 5.0)
            ty_cur = [(-N*sigma, +N*sigma)]

            # Sort upstream masks and attenuators by Z
            masks_up = sorted([m for m in masks if m.z_mm < s.screen_z_mm], key=lambda m: m.z_mm)
            atts_up = sorted([a for a in atts if a.z_mm < s.screen_z_mm], key=lambda a: a.z_mm)

            fig2, (ax_td, ax_sp) = plt.subplots(1, 2, figsize=(12, 5))

            def draw_envelope(ax, intervals, z0, z1, color, label=None):
                if not intervals or z1 <= z0:
                    return
                z = np.linspace(z0, z1, 50)
                for (th_min, th_max) in intervals:
                    y1 = np.tan(th_min) * z
                    y2 = np.tan(th_max) * z
                    ax.fill_between(z, y1, y2, color=color, alpha=0.25, label=label, linewidth=0.0, edgecolor="none", zorder=2)
                    label = None  # only show once

            # Step through segments from source -> mask1 -> mask2 -> ... -> screen
            z_prev = 0.0
            blocked_at = None
            # For visual reference: draw mask apertures and attenuators
            for a in atts_up:
                ax_td.axvline(a.z_mm, color='dimgray', linestyle='--', alpha=0.6)
                ax_sp.axvline(a.z_mm, color='dimgray', linestyle='--', alpha=0.6)
            # Masks drawn later to match each segment end

            for m in masks_up:
                # Draw current envelopes up to this mask
                draw_envelope(ax_td, tx_cur, z_prev, m.z_mm, color='orange', label='Allowed envelope')
                draw_envelope(ax_sp, ty_cur, z_prev, m.z_mm, color='orange', label='Allowed envelope')
                # Draw mask apertures as thick vertical bars at m.z
                # (drawing of mask visuals deferred until after envelopes)
                # Update current angular windows by intersecting with this mask's angular acceptance
                from .masks import mask_angular_windows
                mx, my = mask_angular_windows(m)
                # Intersect; convert any empty/unbounded to empty
                tx_cur = _intersect_unions(tx_cur, mx) if tx_cur and mx else []
                ty_cur = _intersect_unions(ty_cur, my) if ty_cur and my else []
                if not tx_cur or not ty_cur:
                    blocked_at = m
                    z_prev = m.z_mm
                    break
                z_prev = m.z_mm

            # Final segment up to screen
            z_end = s.screen_z_mm
            if not blocked_at:
                draw_envelope(ax_td, tx_cur, z_prev, z_end, color='orange')
                draw_envelope(ax_sp, ty_cur, z_prev, z_end, color='orange')

            # Decorations and labels
            ax_td.axvline(s.screen_z_mm, color='red', linestyle='-', alpha=0.7, label='Screen')
            ax_sp.axvline(s.screen_z_mm, color='red', linestyle='-', alpha=0.7)
            ax_td.set_xlabel('Z [mm]')
            ax_td.set_ylabel('X [mm]')
            ax_td.set_title('Top-down envelope (X vs Z)')
            ax_sp.set_xlabel('Z [mm]')
            ax_sp.set_ylabel('Y [mm]')
            ax_sp.set_title('Side-profile envelope (Y vs Z)')
            ax_td.grid(True, ls=':'); ax_sp.grid(True, ls=':')

            # Draw mask planes as thin black lines with gaps at apertures
            def _merge_intervals_clip(intervals, lo, hi):
                if not intervals:
                    return []
                # clip to [lo, hi]
                clipped = []
                for a, b in intervals:
                    a2, b2 = max(lo, min(a, b)), min(hi, max(a, b))
                    if a2 < b2:
                        clipped.append((a2, b2))
                if not clipped:
                    return []
                clipped.sort(key=lambda x: x[0])
                merged = [clipped[0]]
                for s0, e0 in clipped[1:]:
                    s, e = merged[-1]
                    if s0 <= e:
                        merged[-1] = (s, max(e, e0))
                    else:
                        merged.append((s0, e0))
                return merged

            def _complement_intervals(lo, hi, opens):
                if lo >= hi:
                    return []
                if not opens:
                    return [(lo, hi)]
                opens = _merge_intervals_clip(opens, lo, hi)
                closed = []
                cur = lo
                for s, e in opens:
                    if cur < s:
                        closed.append((cur, s))
                    cur = max(cur, e)
                if cur < hi:
                    closed.append((cur, hi))
                return closed

            # Use current axis limits for visual extent
            td_lo, td_hi = ax_td.get_ylim()
            sp_lo, sp_hi = ax_sp.get_ylim()

            for m in masks_up:
                # Top-down (X vs Z): build open intervals from apertures' X extents
                open_x = []
                for ap in m.apertures:
                    hx = ap.width_mm / 2.0
                    open_x.append((ap.center_x_mm - hx, ap.center_x_mm + hx))
                closed_x = _complement_intervals(td_lo, td_hi, open_x)
                for a, b in closed_x:
                    ax_td.plot([m.z_mm, m.z_mm], [a, b], color='k', linewidth=1.2, alpha=0.9, zorder=4)

                # Side-profile (Y vs Z): build open intervals from apertures' Y extents
                open_y = []
                for ap in m.apertures:
                    hy = ap.height_mm / 2.0
                    open_y.append((ap.center_y_mm - hy, ap.center_y_mm + hy))
                closed_y = _complement_intervals(sp_lo, sp_hi, open_y)
                for a, b in closed_y:
                    ax_sp.plot([m.z_mm, m.z_mm], [a, b], color='k', linewidth=1.2, alpha=0.9, zorder=4)

            # Legends (avoid duplicate labels)
            handles, labels = ax_td.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax_td.legend(by_label.values(), by_label.keys(), loc='best')
            if blocked_at is not None:
                ax_td.text(blocked_at.z_mm, 0.0, f"Blocked at {blocked_at.name}", color='red', fontsize=9, rotation=90, va='bottom', ha='center')
                ax_sp.text(blocked_at.z_mm, 0.0, f"Blocked at {blocked_at.name}", color='red', fontsize=9, rotation=90, va='bottom', ha='center')

            plt.tight_layout()
            plt.show()

            # Persist current scenario
            try:
                self._save_state()
            except Exception:
                pass

            # Summary outputs table
            def _fmtf(v, nd=3):
                try:
                    return f"{float(v):.{nd}f}"
                except Exception:
                    return str(v)
            def _fmt_sig(v, sig=6):
                try:
                    return format(float(v), f'.{sig}g')
                except Exception:
                    return str(v)
            def _bounds(win):
                try:
                    if not win:
                        return (None, None)
                    lo = min(w[0] for w in win)
                    hi = max(w[1] for w in win)
                    return (lo, hi)
                except Exception:
                    return (None, None)
            tx_lo, tx_hi = _bounds(res.theta_x_windows_rad)
            ty_lo, ty_hi = _bounds(res.theta_y_windows_rad)
            magnet_title = self.dd_set.value
            n_masks_up = len([m for m in masks if m.z_mm < s.screen_z_mm])
            n_atts_up = len([a for a in atts if a.z_mm < s.screen_z_mm])
            n_fps = len(fps) if 'fps' in locals() else 0
            sum_fp_W = sum(fp.total_power_W for fp in fps) if 'fps' in locals() and fps else 0.0
            atten_status = "Yes" if xraylib is not None else "No (disabled)"
            # Sigma and P_peak per provided formulas
            gamma = 1957.0 * s.energy_GeV
            sigma_rad = 0.608 / gamma
            sigma_mrad = sigma_rad * 1e3
            ppk_parts = []
            for _mag in mset.magnets:
                _ppk = 5.42 * _mag.B_T * (s.energy_GeV ** 4) * s.current_A
                ppk_parts.append(f"{_mag.name}: {_fmtf(_ppk, 3)}")
            ppeak_html = "; ".join(ppk_parts)
            rows = []
            rows.append(f"<tr><th>Magnet set</th><td>{magnet_title}</td></tr>")
            rows.append(f"<tr><th>Screen Z [mm]</th><td>{_fmtf(s.screen_z_mm, 3)}</td></tr>")
            rows.append(f"<tr><th>Sigma [mrad]</th><td>{_fmt_sig(sigma_mrad, 6)}</td></tr>")
            rows.append(f"<tr><th>P_peak (5.42·B·E^4·I) [W]</th><td>{ppeak_html}</td></tr>")
            rows.append(f"<tr><th>Total power at screen [W]</th><td>{_fmtf(res.total_power_W, 3)}</td></tr>")

            summary_html = (
                "<div style=\"margin-top:10px\">"
                "<h4 style=\"margin:4px 0\">Run Summary</h4>"
                "<table>" + "".join(rows) + "</table>"
                "</div>"
            )
            display(widgets.HTML(summary_html))
            # Per-magnet subtable
            gamma = 1957.0 * s.energy_GeV
            sigma_rad = 0.608 / gamma
            sigma_mrad = sigma_rad * 1e3
            downstream_mag = max(mset.magnets, key=lambda m: m.z_mm) if mset.magnets else None

            def _intersect_windows_half_open(windows, th_lo, th_hi, include_upper):
                out = []
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

            rows_mag = []
            rows_mag.append(
                "<tr><th>Magnet</th><th>B [T]</th><th>theta min [mrad]</th><th>theta max [mrad]</th><th>z [mm]</th><th>Sigma [mrad]</th><th>P_peak [W]</th><th>Power [W]</th><th>Ltheta [W/mrad]</th><th>Lx [W/mm]</th><th>E_c [keV]</th></tr>"
            )
            gamma = 1957.0 * s.energy_GeV
            sigma_mrad = (0.608 / gamma) * 1e3
            downstream_mag = max(mset.magnets, key=lambda m: m.z_mm) if mset.magnets else None
            def _intersect_windows_half_open(windows, th_lo, th_hi, include_upper):
                out = []
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
            for mag in mset.magnets:
                include_upper = (mag is downstream_mag)
                th_lo = min(mag.theta_min_mrad, mag.theta_max_mrad) * 1e-3
                th_hi = max(mag.theta_min_mrad, mag.theta_max_mrad) * 1e-3
                masked_theta = _intersect_windows_half_open(res.theta_x_windows_rad, th_lo, th_hi, include_upper)
                width_mrad = 0.0
                for s_th, e_th in masked_theta:
                    width_mrad += (e_th - s_th) * 1e3
                Ltheta_mrad = 4.22 * mag.B_T * (s.energy_GeV ** 3) * s.current_A
                z_mag_mm = (s.screen_z_mm - mag.z_mm)
                Lx_mm = (Ltheta_mrad * (1000.0 / z_mag_mm)) if z_mag_mm > 0 else 0.0
                Pmag_W = Ltheta_mrad * width_mrad
                Ppeak_W = 5.42 * mag.B_T * (s.energy_GeV ** 4) * s.current_A
                Ec_keV = 0.665 * (s.energy_GeV ** 2) * mag.B_T
                rows_mag.append(
                    f"<tr><td>{mag.name}</td><td>{_fmtf(mag.B_T, 3)}</td><td>{_fmtf(mag.theta_min_mrad, 3)}</td><td>{_fmtf(mag.theta_max_mrad, 3)}</td><td>{_fmtf(mag.z_mm, 3)}</td><td>{_fmt_sig(sigma_mrad, 6)}</td><td>{_fmtf(Ppeak_W, 3)}</td><td>{_fmtf(Pmag_W, 3)}</td><td>{_fmtf(Ltheta_mrad, 3)}</td><td>{_fmtf(Lx_mm, 3)}</td><td>{_fmtf(Ec_keV, 3)}</td></tr>"
                )
            mag_html = (
                "<div style=\"margin-top:10px\">"
                "<h4 style=\"margin:4px 0\">Per-Magnet Summary</h4>"
                "<table>" + "".join(rows_mag) + "</table>"
                "</div>"
            )
            display(widgets.HTML(mag_html))

    def _state_path(self) -> str:
        return os.path.join(os.getcwd(), "bm_power_state.json")

    def _serialize(self) -> dict:
        # Settings
        cfg = {
            "magnet_set_title": self.dd_set.value,
            "settings": {
                "energy_GeV": self.energy.value,
                "current_A": self.current.value,
                "screen_z_mm": self.screen_z.value,
                "nx": self.nx.value,
                "ny": self.ny.value,
                "Emin_eV": self.Emin.value,
                "Emax_eV": self.Emax.value,
                "nE": self.nE.value,
                "logE": bool(self.logE.value),
            },
            "masks": [
                {
                    "name": mw.name.value,
                    "z_mm": mw.z.value,
                    "apertures": [
                        {
                            "width_mm": apw.w_mm.value,
                            "height_mm": apw.h_mm.value,
                            "center_x_mm": apw.cx.value,
                            "center_y_mm": apw.cy.value,
                        }
                        for apw in mw.apertures
                    ],
                }
                for mw in self.masks_widgets
            ],
            "attenuators": [
                {
                    "z_mm": aw.z.value,
                    "material": aw.material.value,
                    "thickness_mm": aw.thickness.value,
                }
                for aw in self.att_widgets
            ],
        }
        return cfg

    def _apply_state(self, state: dict):
        try:
            # Magnet set
            title = state.get("magnet_set_title")
            if title in self.magnet_sets:
                self.dd_set.value = title
            # Settings
            st = state.get("settings", {})
            if st:
                self.energy.value = st.get("energy_GeV", self.energy.value)
                self.current.value = st.get("current_A", self.current.value)
                self.screen_z.value = st.get("screen_z_mm", self.screen_z.value)
                self.nx.value = st.get("nx", self.nx.value)
                self.ny.value = st.get("ny", self.ny.value)
                self.Emin.value = st.get("Emin_eV", self.Emin.value)
                self.Emax.value = st.get("Emax_eV", self.Emax.value)
                self.nE.value = st.get("nE", self.nE.value)
                self.logE.value = bool(st.get("logE", self.logE.value))
            # Masks
            masks_state = state.get("masks", [])
            if masks_state:
                new_mws: List[MaskWidget] = []
                for i, m in enumerate(masks_state, start=1):
                    aps = [Aperture(**ap) for ap in m.get("apertures", [])]
                    mm = Mask(name=m.get("name", f"Mask{i}"), z_mm=m.get("z_mm", 0.0), apertures=aps)
                    new_mws.append(MaskWidget(i, mm, on_remove=self._on_remove_mask))
                self.masks_widgets = new_mws
                self.masks_container.children = [mw.w for mw in self.masks_widgets]
            # Attenuators
            atts_state = state.get("attenuators", [])
            if atts_state:
                new_aws: List[AttenuatorWidget] = []
                for i, a in enumerate(atts_state, start=1):
                    aw = AttenuatorWidget(i, Attenuator(z_mm=a.get("z_mm", 0.0), material=a.get("material", materials[0]), thickness_mm=a.get("thickness_mm", 0.1)), materials, on_remove=self._on_remove_att)
                    new_aws.append(aw)
                self.att_widgets = new_aws
                self.atts_container.children = [aw.w for aw in self.att_widgets]
        except Exception as e:
            print("WARNING: Failed to apply saved state:", e)

    def _load_state(self):
        try:
            path = self._state_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self._apply_state(state)
        except Exception as e:
            print("WARNING: Failed to load saved state:", e)

    def _save_state(self):
        try:
            path = self._state_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._serialize(), f, indent=2)
        except Exception as e:
            print("WARNING: Failed to save state:", e)

    def display(self):
        display(self.root)

def launch():
    ui = BeamlineUI()
    ui.display()
    return ui



