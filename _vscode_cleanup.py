import io, os, re

root = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src"

def edit_file(path, editor):
    with io.open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    new = editor(txt)
    if new != txt:
        with io.open(path, "w", encoding="utf-8") as f:
            f.write(new)
        print(f"Updated {os.path.basename(path)}")
    else:
        print(f"No changes needed in {os.path.basename(path)}")

# --- compute.py: remove unused imports ---
def fix_compute(txt: str) -> str:
    # Remove get_available_magnet_sets from magnets import
    txt = re.sub(r"(from\s+\.magnets\s+import\s+[^,\n]*)(,\s*get_available_magnet_sets)", r"\1", txt)
    # Remove filter_upstream and transmission_curve from attenuation import
    txt = re.sub(r"(from\s+\.attenuation\s+import\s+Attenuator)(,\s*filter_upstream)?(,\s*transmission_curve)?", r"\1", txt)
    # Remove any lingering XrtBendingMagnet references
    txt = re.sub(r".*XrtBendingMagnet.*\n?", "", txt)
    # Tidy excessive blank lines
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt

# --- attenuation.py: remove unused Tuple import ---
def fix_attenuation(txt: str) -> str:
    txt = re.sub(r"from\s+typing\s+import\s+([^#\n]*)", lambda m: m.group(0).replace("Tuple", "").replace(",  ", ", ").replace(",,", ","), txt)
    # Clean trailing commas/spaces
    txt = re.sub(r"from\s+typing\s+import\s+,?\s*", "from typing import ", txt)
    txt = re.sub(r"from\s+typing\s+import\s+\s*$", "from typing import Dict, List", txt)
    return txt

# --- footprints.py: remove unused Optional import ---
def fix_footprints(txt: str) -> str:
    txt = re.sub(r"from\s+typing\s+import\s+([^#\n]*)", lambda m: m.group(0).replace("Optional, ", "").replace(", Optional", ""), txt)
    return txt

# --- ui_widgets.py: cleanup imports and duplicate def, fix multiple imports ---
def fix_ui_widgets(txt: str) -> str:
    # Remove unused imports
    txt = re.sub(r"from\s+dataclasses\s+import\s+dataclass\s*\n", "", txt)
    txt = re.sub(r"from\s+typing\s+import\s+List,\s*Optional,\s*Dict\s*\n", "from typing import List\n", txt)
    txt = re.sub(r"from\s+matplotlib\.patches\s+import\s+Rectangle\s*\n", "", txt)
    # Split 'import json, os' into two lines
    txt = re.sub(r"^import\s+json,\s*os\s*$", "import json\nimport os", txt, flags=re.M)
    # Remove default_masks and default_attenuators from ui_presets import (keep what is used)
    txt = re.sub(r"from\s+\.ui_presets\s+import\s*\(\s*[^)]*\)", lambda m: m.group(0).replace("default_masks,", "").replace("default_attenuators,", ""), txt)
    # Remove duplicate _intersect_windows_half_open definition in per-magnet block (keep first one)
    blocks = re.findall(r"(?ms)^def\s+_intersect_windows_half_open\s*\(.*?\)\s*:\s*.*?(?=^def\s|^for\s|^rows_mag|^display|$)", txt)
    if len(blocks) > 1:
        # Remove all but the first occurrence
        first = blocks[0]
        rest = blocks[1:]
        for b in rest:
            txt = txt.replace(b, "")
    # Remove unused variable assignments like 'materials = ...' inside _on_add_att if not used
    txt = re.sub(r"^\s*materials\s*=\s*materials_dropdown_options\(\)\s*\n", "", txt, flags=re.M)
    # Replace 'except Exception as _:' with 'except Exception:'
    txt = re.sub(r"except\s+Exception\s+as\s+_\s*:", "except Exception:", txt)
    # Tidy multiple statements on one line by replacing semicolons in labels line (minor)
    txt = re.sub(r"(\s*ax_td\.set_xlabel\('Z \[mm\]'\));\s*ax_td\.set_ylabel\('X \[mm\]'\);\s*ax_td\.set_title\('Top-down envelope \(X vs Z\)'\)", r"ax_td.set_xlabel('Z [mm]')\n            ax_td.set_ylabel('X [mm]')\n            ax_td.set_title('Top-down envelope (X vs Z)')", txt)
    txt = re.sub(r"(\s*ax_sp\.set_xlabel\('Z \[mm\]'\));\s*ax_sp\.set_ylabel\('Y \[mm\]'\);\s*ax_sp\.set_title\('Side-profile envelope \(Y vs Z\)'\)", r"ax_sp.set_xlabel('Z [mm]')\n            ax_sp.set_ylabel('Y [mm]')\n            ax_sp.set_title('Side-profile envelope (Y vs Z)')", txt)
    # Tidy excessive blank lines
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt

edit_file(os.path.join(root, "compute.py"), fix_compute)
edit_file(os.path.join(root, "attenuation.py"), fix_attenuation)
edit_file(os.path.join(root, "footprints.py"), fix_footprints)
edit_file(os.path.join(root, "ui_widgets.py"), fix_ui_widgets)

print("Cleanup edits applied.")