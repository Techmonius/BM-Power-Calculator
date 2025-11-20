import io, re, sys

path = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src\ui_widgets.py"

with io.open(path, "r", encoding="utf-8") as f:
    txt = f.read()

lines = txt.splitlines()

def replace_in_lines(predicate, replacer):
    changed = False
    for i, line in enumerate(lines):
        if predicate(line):
            new_line = replacer(line)
            if new_line != line:
                lines[i] = new_line
                changed = True
    return changed

# 1) Footprints header labels to ASCII (Width theta / Height psi)
txt = "\n".join(lines)
txt = txt.replace("Width ? [mrad]", "Width theta [mrad]")
txt = txt.replace("Height ? [mrad]", "Height psi [mrad]")
txt = txt.replace("Width θ [mrad]", "Width theta [mrad]")
txt = txt.replace("Height ψ [mrad]", "Height psi [mrad]")
lines = txt.splitlines()

# 2) Footprint number column left-aligned (cell with {fp.label})
def is_fp_label_cell(line: str) -> bool:
    return ("{fp.label}" in line) and ("<td" in line)

def left_align_td(line: str) -> str:
    return line.replace("text-align:right", "text-align:left")

replace_in_lines(is_fp_label_cell, left_align_td)

# 3) Per-magnet Magnet name column left-aligned (cell with {mag.name})
def is_mag_name_cell(line: str) -> bool:
    return ("{mag.name}" in line) and ("<td" in line)

replace_in_lines(is_mag_name_cell, left_align_td)

# 4) Run Summary: left-align value cell for Magnet set and P_peak rows
def is_magset_row(line: str) -> bool:
    return ("<th" in line and ">Magnet set<" in line and "<td" in line)

def is_ppeak_row(line: str) -> bool:
    return ("<th" in line and ">P_peak" in line and "<td" in line)

replace_in_lines(is_magset_row, left_align_td)
replace_in_lines(is_ppeak_row, left_align_td)

# Write back
new_txt = "\n".join(lines)
with io.open(path, "w", encoding="utf-8") as f:
    f.write(new_txt)

print("Applied alignment and label fixes.")
