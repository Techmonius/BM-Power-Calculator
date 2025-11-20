import re

path = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src\ui_widgets.py"

with open(path, "r", encoding="utf-8") as f:
    txt = f.read()

# 1) Add Lx [W/mm] to per-magnet header (insert after Ltheta [W/mrad])
header_insert = "<th style='padding:6px 10px;border:1px solid #ddd;background:#f7f7f7;text-align:left'>Lx [W/mm]</th>"
txt = txt.replace("Ltheta [W/mrad]</th>", "Ltheta [W/mrad]</th>" + header_insert)

# 2) Insert Lx_mm computation right after Ltheta_mrad assignment
lines = txt.splitlines()
for i, line in enumerate(lines):
    if ("Ltheta_mrad =" in line) and ("4.22" in line) and ("B_T" in line) and ("energy_GeV" in line):
        indent = line[:len(line)-len(line.lstrip())]
        # Check if next line already defines Lx_mm
        if i+1 < len(lines) and ("Lx_mm =" in lines[i+1]):
            pass
        else:
            lines.insert(i+1, indent + "Lx_mm = Ltheta_mrad * (1000.0 / s.screen_z_mm)")
        break
txt = "\n".join(lines)

# 3) Insert Lx cell before Ec_keV cell in per-magnet rows
lines = txt.splitlines()
for i, line in enumerate(lines):
    if ("_fmtf(Ec_keV, 3)" in line) and ("<td" in line) and ("rows_mag.append" in "".join(lines[max(0,i-10):i+1])):
        indent = line[:len(line)-len(line.lstrip())]
        lx_cell = indent + "f\"<td style='padding:6px 10px;border:1px solid #ddd;text-align:right'>{_fmtf(Lx_mm, 3)}</td>\""
        # Avoid duplicate insertion
        prev_block = "\n".join(lines[max(0, i-5):i])
        if "_fmtf(Lx_mm, 3)" not in prev_block:
            lines.insert(i, lx_cell)
        break

txt = "\n".join(lines)

with open(path, "w", encoding="utf-8") as f:
    f.write(txt)

print("Applied per-magnet Lx column and value cell.")