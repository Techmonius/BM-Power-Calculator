import io, os, re

root = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src"
att_path = os.path.join(root, "attenuation.py")
uiw_path = os.path.join(root, "ui_widgets.py")

# 1) Fix trailing comma in typing import in attenuation.py
with io.open(att_path, "r", encoding="utf-8") as f:
    att = f.read()
# Remove trailing comma at end of typing import line
att = re.sub(r"^(from\s+typing\s+import\s+.*?),\s*$", r"\1", att, flags=re.M)
with io.open(att_path, "w", encoding="utf-8") as f:
    f.write(att)
print("Fixed attenuation.py import line.")

# 2) Fix jammed statements in ui_widgets.py (split ax_sp.axvline and ax_td.set_xlabel)
with io.open(uiw_path, "r", encoding="utf-8") as f:
    uiw = f.read()
uiw_new = uiw
# Split '... )ax_td.set_xlabel('Z [mm]')' into two lines
uiw_new = re.sub(
    r"\)\s*ax_td\.set_xlabel\('Z \[mm\]'\)",
    ")\n            ax_td.set_xlabel('Z [mm]')",
    uiw_new,
)
# Also ensure ax_td's paired ylabel/title lines are on separate lines (if semicolons remain)
uiw_new = re.sub(
    r"ax_td\.set_xlabel\('Z \[mm\]'\);\s*ax_td\.set_ylabel\('X \[mm\]'\);\s*ax_td\.set_title\('Top-down envelope \(X vs Z\)'\)",
    "ax_td.set_xlabel('Z [mm]')\n            ax_td.set_ylabel('X [mm]')\n            ax_td.set_title('Top-down envelope (X vs Z)')",
    uiw_new,
)
uiw_new = re.sub(
    r"ax_sp\.set_xlabel\('Z \[mm\]'\);\s*ax_sp\.set_ylabel\('Y \[mm\]'\);\s*ax_sp\.set_title\('Side-profile envelope \(Y vs Z\)'\)",
    "ax_sp.set_xlabel('Z [mm]')\n            ax_sp.set_ylabel('Y [mm]')\n            ax_sp.set_title('Side-profile envelope (Y vs Z)')",
    uiw_new,
)
if uiw_new != uiw:
    with io.open(uiw_path, "w", encoding="utf-8") as f:
        f.write(uiw_new)
    print("Fixed ui_widgets.py jammed statements.")
else:
    print("No jammed statements found to fix in ui_widgets.py.")