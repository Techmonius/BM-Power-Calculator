import io, re

path = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src\ui_widgets.py"

with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find Per-Magnet Summary block from rows_mag = [] to display(widgets.HTML(mag_html))
start = -1
end = -1
for i, ln in enumerate(lines):
    if start < 0 and re.search(r"^\s*rows_mag\s*=\s*\[\]", ln):
        start = i
    if start >= 0 and re.search(r"^\s*display\(widgets\.HTML\(mag_html\)\)", ln):
        end = i
        break

if start >= 0 and end > start:
    indent = re.match(r"(\s*)", lines[start]).group(1)
    block = [
        indent + "rows_mag = []\n",
        indent + "rows_mag.append(\n",
        indent + "    \"<tr><th>Magnet</th><th>Sigma [mrad]</th><th>P_peak [W]</th><th>Power [W]</th><th>Ltheta [W/mrad]</th><th>Lx [W/mm]</th><th>E_c [keV]</th></tr>\"\n",
        indent + ")\n",
        indent + "gamma = 1957.0 * s.energy_GeV\n",
        indent + "sigma_mrad = (0.608 / gamma) * 1e3\n",
        indent + "downstream_mag = max(mset.magnets, key=lambda m: m.z_mm) if mset.magnets else None\n",
        indent + "def _intersect_windows_half_open(windows, th_lo, th_hi, include_upper):\n",
        indent + "    out = []\n",
        indent + "    lo = min(th_lo, th_hi)\n",
        indent + "    hi = max(th_lo, th_hi)\n",
        indent + "    for a, b in (windows or []):\n",
        indent + "        s = max(a, lo)\n",
        indent + "        e = min(b, hi)\n",
        indent + "        if include_upper:\n",
        indent + "            if s <= e:\n",
        indent + "                out.append((s, e))\n",
        indent + "        else:\n",
        indent + "            if s < e:\n",
        indent + "                out.append((s, e))\n",
        indent + "    return out\n",
        indent + "for mag in mset.magnets:\n",
        indent + "    include_upper = (mag is downstream_mag)\n",
        indent + "    th_lo = min(mag.theta_min_mrad, mag.theta_max_mrad) * 1e-3\n",
        indent + "    th_hi = max(mag.theta_min_mrad, mag.theta_max_mrad) * 1e-3\n",
        indent + "    masked_theta = _intersect_windows_half_open(res.theta_x_windows_rad, th_lo, th_hi, include_upper)\n",
        indent + "    width_mrad = 0.0\n",
        indent + "    for s_th, e_th in masked_theta:\n",
        indent + "        width_mrad += (e_th - s_th) * 1e3\n",
        indent + "    Ltheta_mrad = 4.22 * mag.B_T * (s.energy_GeV ** 3) * s.current_A\n",
        indent + "    z_mag_mm = (s.screen_z_mm - mag.z_mm)\n",
        indent + "    Lx_mm = (Ltheta_mrad * (1000.0 / z_mag_mm)) if z_mag_mm > 0 else 0.0\n",
        indent + "    Pmag_W = Ltheta_mrad * width_mrad\n",
        indent + "    Ppeak_W = 5.42 * mag.B_T * (s.energy_GeV ** 4) * s.current_A\n",
        indent + "    Ec_keV = 0.665 * (s.energy_GeV ** 2) * mag.B_T\n",
        indent + "    rows_mag.append(\n",
        indent + "        f\"<tr><td>{mag.name}</td><td>{_fmt_sig(sigma_mrad, 6)}</td><td>{_fmtf(Ppeak_W, 3)}</td><td>{_fmtf(Pmag_W, 3)}</td><td>{_fmtf(Ltheta_mrad, 3)}</td><td>{_fmtf(Lx_mm, 3)}</td><td>{_fmtf(Ec_keV, 3)}</td></tr>\"\n",
        indent + "    )\n",
        indent + "mag_html = (\n",
        indent + "    \"<div style=\\\"margin-top:10px\\\">\"\n",
        indent + "    \"<h4 style=\\\"margin:4px 0\\\">Per-Magnet Summary</h4>\"\n",
        indent + "    \"<table>\" + \"\".join(rows_mag) + \"</table>\"\n",
        indent + "    \"</div>\"\n",
        indent + ")\n",
        indent + "display(widgets.HTML(mag_html))\n",
    ]
    # Replace the block
    lines = lines[:start] + block + lines[end+1:]
    changed = True
else:
    changed = False

with io.open(path, "w", encoding="utf-8") as f:
    f.write("".join(lines))

print("Per-Magnet Summary block rebuilt.")