import io, re

path = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src\ui_widgets.py"

with io.open(path, "r", encoding="utf-8") as f:
    txt = f.read()

lines = txt.splitlines()

# 1) Fix Lx [W/mm] to use per-magnet distance to screen: z_mag_mm = screen_z - mag.z_mm
for i, line in enumerate(lines):
    if "Lx_mm = Ltheta_mrad * (1000.0 / s.screen_z_mm)" in line:
        indent = line[:len(line)-len(line.lstrip())]
        # Replace with z_mag distance handling
        lines[i] = indent + "z_mag_mm = (s.screen_z_mm - mag.z_mm)"
        lines.insert(i+1, indent + "Lx_mm = (Ltheta_mrad * (1000.0 / z_mag_mm)) if z_mag_mm > 0 else 0.0")
        break

# 2) Make Per-Magnet Summary table span full width: set table style width:100%
# Find mag_html assignment block and replace <table> with width:100%
for i, line in enumerate(lines):
    if "mag_html = (" in line:
        # Search forward a few lines for the table tag concatenation
        for j in range(i, min(i+20, len(lines))):
            if "<table" in lines[j]:
                lines[j] = re.sub(r'<table[^>]*>', '<table style=\"width:100%\">', lines[j])
                break
        break

new_txt = "\n".join(lines)
with io.open(path, "w", encoding="utf-8") as f:
    f.write(new_txt)

print("Updated Lx to use per-magnet z and set Per-Magnet table width to 100%.")