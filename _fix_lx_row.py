import io

path = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src\ui_widgets.py"

with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the rows_mag.append block that currently adds an Lx <td> on its own line, then a full <tr>
start = -1
end = -1
for i, ln in enumerate(lines):
    if start == -1 and "rows_mag.append(" in ln:
        start = i
        # look ahead for the closing ')'
        for j in range(i, min(i+20, len(lines))):
            if ")" in lines[j]:
                end = j
                break
        break

if start != -1 and end != -1:
    # Determine indentation
    indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    # Replace the whole block with a single row including Lx in the same <tr>
    new_block = [
        indent + "rows_mag.append(",
        indent + "    f\"<tr><td>{mag.name}</td><td>{_fmt_sig(sigma_mrad, 6)}</td><td>{_fmtf(Ppeak_W, 3)}</td><td>{_fmtf(Pmag_W, 3)}</td><td>{_fmtf(Ltheta_mrad, 3)}</td><td>{_fmtf(Lx_mm, 3)}</td><td>{_fmtf(Ec_keV, 3)}</td></tr>\"",
        indent + ")",
    ]
    lines = lines[:start] + new_block + lines[end+1:]
    changed = True
else:
    changed = False

with io.open(path, "w", encoding="utf-8") as f:
    f.write("".join(lines))

print("Rebuilt per-magnet row to include Lx in the same <tr>.")