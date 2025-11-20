import io

path = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src\ui_widgets.py"

with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def style_cell(line, align):
    # Ensure a <td> has inline style with desired alignment, preserving any existing padding/border
    if "<td" not in line:
        return line
    if "style=" in line:
        # flip alignment inside style
        if "text-align:right" in line and align == "left":
            return line.replace("text-align:right", "text-align:left")
        if "text-align:left" in line and align == "right":
            return line.replace("text-align:left", "text-align:right")
        # if style exists but no text-align, inject it before closing quote
        parts = line.split("style='")
        if len(parts) == 2 and "text-align" not in line:
            before, after = parts[0], parts[1]
            after = "text-align:{};".format(align) + after
            return before + "style='" + after
        parts = line.split('style="')
        if len(parts) == 2 and "text-align" not in line:
            before, after = parts[0], parts[1]
            after = "text-align:{};".format(align) + after
            return before + 'style="' + after
        return line
    else:
        # insert a style attribute with padding/border and alignment
        return line.replace("<td", "<td style='padding:6px 10px;border:1px solid #ddd;text-align:{}'".format(align))

def style_th_header(line):
    # Ensure th has header styling and left alignment
    if "<th" in line:
        if "style=" in line:
            # ensure left alignment
            line = line.replace("text-align:right", "text-align:left")
            if "text-align" not in line:
                # inject
                line = line.replace("style='", "style='text-align:left;")
                line = line.replace('style="', 'style="text-align:left;')
            # ensure padding/border/bg
            if "padding:" not in line or "border:" not in line or "background:" not in line:
                # append missing pieces
                if "style='" in line:
                    line = line.replace("style='", "style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;")
                else:
                    line = line.replace('style="', 'style="padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;')
            return line
        else:
            return line.replace("<th", "<th style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;text-align:left'")
    return line

new_lines = []
for i, line in enumerate(lines):
    L = line

    # Footprints header labels to ASCII
    if "Width θ [mrad]" in L or "Width ? [mrad]" in L:
        L = L.replace("Width θ [mrad]", "Width theta [mrad]").replace("Width ? [mrad]", "Width theta [mrad]")
    if "Height ψ [mrad]" in L or "Height ? [mrad]" in L:
        L = L.replace("Height ψ [mrad]", "Height psi [mrad]").replace("Height ? [mrad]", "Height psi [mrad]")

    # Style all <th> header cells left-aligned with padding/border/bg
    if "<tr><th" in L or "<th>" in L:
        L = style_th_header(L)

    # Footprints table row cells: left-align label, right-align numerics
    if "{fp.label}" in L and "<td" in L:
        L = style_cell(L, "left")
    if ("{_fmt(fp.width_x_mm)}" in L or "{_fmt(fp.height_y_mm)}" in L or
        "{_fmt(fp.width_theta_mrad)}" in L or "{_fmt(fp.height_psi_mrad)}" in L or
        "{_fmt(fp.total_power_W)}" in L) and "<td" in L:
        L = style_cell(L, "right")

    # Per-magnet table row cells: left-align magnet name, right-align numeric
    if "{mag.name}" in L and "<td" in L:
        L = style_cell(L, "left")
    if ("{_fmt_sig(sigma_mrad, 6)}" in L or "{_fmtf(Ppeak_W, 3)}" in L or "{_fmtf(Pmag_W, 3)}" in L or
        "{_fmtf(Ltheta_mrad, 3)}" in L or "{_fmtf(Lx_mm, 3)}" in L or "{_fmtf(Ec_keV, 3)}" in L) and "<td" in L:
        L = style_cell(L, "right")

    # Run Summary: left-align Magnet set and P_peak value cells
    if ("<th style='text-align:left;'>Magnet set</th>" in L or "<th style='text-align:left;'>Magnet set</th>" in L) and "<td" in L:
        L = style_cell(L, "left")
    if ("<th style='text-align:left;'>P_peak" in L) and "<td" in L:
        L = style_cell(L, "left")

    new_lines.append(L)

with io.open(path, "w", encoding="utf-8") as f:
    f.write("".join(new_lines))

print("Applied precise per-cell alignment.")