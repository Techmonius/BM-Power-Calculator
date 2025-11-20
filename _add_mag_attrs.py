import io

path = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src\ui_widgets.py"

with io.open(path, "r", encoding="utf-8") as f:
    txt = f.read()

# 1) Update the Per-Magnet Summary header to include magnet attributes: B [T], theta min/max [mrad], z [mm]
old_header = "<tr><th>Magnet</th><th>Sigma [mrad]</th><th>P_peak [W]</th><th>Power [W]</th><th>Ltheta [W/mrad]</th><th>Lx [W/mm]</th><th>E_c [keV]</th></tr>"
new_header = (
    "<tr>"
    "<th>Magnet</th>"
    "<th>B [T]</th>"
    "<th>theta min [mrad]</th>"
    "<th>theta max [mrad]</th>"
    "<th>z [mm]</th>"
    "<th>Sigma [mrad]</th>"
    "<th>P_peak [W]</th>"
    "<th>Power [W]</th>"
    "<th>Ltheta [W/mrad]</th>"
    "<th>Lx [W/mm]</th>"
    "<th>E_c [keV]</th>"
    "</tr>"
)
if old_header in txt:
    txt = txt.replace(old_header, new_header)

# 2) Update the per-magnet row to include those attributes in the same row
old_row = (
    '        f"<tr><td>{mag.name}</td><td>{_fmt_sig(sigma_mrad, 6)}</td><td>{_fmtf(Ppeak_W, 3)}</td>'
    '<td>{_fmtf(Pmag_W, 3)}</td><td>{_fmtf(Ltheta_mrad, 3)}</td><td>{_fmtf(Lx_mm, 3)}</td>'
    '<td>{_fmtf(Ec_keV, 3)}</td></tr>"'
)

# Build a robust replacement even if whitespace differs
if old_row not in txt:
    # Try to find the row append line and replace it by pattern
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        if "rows_mag.append(" in line:
            # Capture block lines for row append
            # Find the line that contains the f-string with <tr><td>{mag.name}
            blk_end = i
            while blk_end < len(lines) and ")" not in lines[blk_end]:
                blk_end += 1
            block = "\n".join(lines[i:blk_end+1])
            if "<tr><td>{mag.name}</td>" in block and "_fmtf(Lx_mm, 3)" in block:
                # Replace the f-string row
                indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
                new_row_line = (
                    indent + '        f"<tr><td>{mag.name}</td>'
                    '<td>{_fmtf(mag.B_T, 3)}</td>'
                    '<td>{_fmtf(mag.theta_min_mrad, 3)}</td>'
                    '<td>{_fmtf(mag.theta_max_mrad, 3)}</td>'
                    '<td>{_fmtf(mag.z_mm, 3)}</td>'
                    '<td>{_fmt_sig(sigma_mrad, 6)}</td>'
                    '<td>{_fmtf(Ppeak_W, 3)}</td>'
                    '<td>{_fmtf(Pmag_W, 3)}</td>'
                    '<td>{_fmtf(Ltheta_mrad, 3)}</td>'
                    '<td>{_fmtf(Lx_mm, 3)}</td>'
                    '<td>{_fmtf(Ec_keV, 3)}</td></tr>"'
                )
                # Rebuild the block with the single-line f-string row
                new_block_lines = []
                in_append = False
                for j in range(i, blk_end+1):
                    if "rows_mag.append(" in lines[j] and not in_append:
                        in_append = True
                        new_block_lines.append(lines[j])  # keep the "rows_mag.append(" line
                        new_block_lines.append(new_row_line)
                    elif ")" in lines[j] and in_append:
                        new_block_lines.append(lines[j])  # closing parenthesis
                        in_append = False
                    # else skip intermediate lines
                # Replace original lines
                lines[i:blk_end+1] = new_block_lines
                txt = "\n".join(lines)
            break
else:
    # Straight replacement path (if exact old row string exists)
    new_row = (
        '        f"<tr><td>{mag.name}</td>'
        '<td>{_fmtf(mag.B_T, 3)}</td>'
        '<td>{_fmtf(mag.theta_min_mrad, 3)}</td>'
        '<td>{_fmtf(mag.theta_max_mrad, 3)}</td>'
        '<td>{_fmtf(mag.z_mm, 3)}</td>'
        '<td>{_fmt_sig(sigma_mrad, 6)}</td>'
        '<td>{_fmtf(Ppeak_W, 3)}</td>'
        '<td>{_fmtf(Pmag_W, 3)}</td>'
        '<td>{_fmtf(Ltheta_mrad, 3)}</td>'
        '<td>{_fmtf(Lx_mm, 3)}</td>'
        '<td>{_fmtf(Ec_keV, 3)}</td></tr>"'
    )
    txt = txt.replace(old_row, new_row)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(txt)

print("Added magnet attributes columns to Per-Magnet Summary.")