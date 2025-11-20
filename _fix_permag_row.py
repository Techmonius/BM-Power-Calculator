import io, re

path = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src\ui_widgets.py"

with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 1) Fix per-magnet row so Lx is in the same <tr> (remove stray <td> before the row)
for idx in range(len(lines)):
    if "rows_mag.append(" in lines[idx]:
        # find end of this append block
        end = idx
        while end < len(lines) and ")" not in lines[end]:
            end += 1
        block_text = "".join(lines[idx:end+1])
        # Only rewrite if we see the stray Lx cell and the row in this block
        if "_fmtf(Lx_mm" in block_text and "<tr><td>{mag.name}" in block_text:
            indent = re.match(r"(\s*)", lines[idx]).group(1)
            new_block = [
                indent + "rows_mag.append(\n",
                indent + "    f\"<tr><td>{mag.name}</td><td>{_fmt_sig(sigma_mrad, 6)}</td><td>{_fmtf(Ppeak_W, 3)}</td><td>{_fmtf(Pmag_W, 3)}</td><td>{_fmtf(Ltheta_mrad, 3)}</td><td>{_fmtf(Lx_mm, 3)}</td><td>{_fmtf(Ec_keV, 3)}</td></tr>\"\n",
                indent + ")\n",
            ]
            lines[idx:end+1] = new_block
        break

# 2) Fix per-magnet table width quoting: "<table style=\"width:100%\">"
for i, line in enumerate(lines):
    if '<table style="width:100%">' in line:
        lines[i] = line.replace('<table style="width:100%">', '<table style=\\"width:100%\\">')

with io.open(path, "w", encoding="utf-8") as f:
    f.write("".join(lines))

print("Fixed per-magnet rows (Lx in same row) and table width quoting.")