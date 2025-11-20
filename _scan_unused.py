import os, re, io

root = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src"

def read_text(path):
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

py_files = []
for dirpath, dirnames, filenames in os.walk(root):
    for fn in filenames:
        if fn.endswith(".py"):
            py_files.append(os.path.join(dirpath, fn))

# Collect defs and simple identifiers per file
defs_by_file = {}
idents_by_file = {}
for f in py_files:
    txt = read_text(f)
    defs = []
    # function and class defs
    for m in re.finditer(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', txt, flags=re.M):
        defs.append(m.group(1))
    for m in re.finditer(r'^\s*class\s+([A-Za-z_]\w*)\s*\(', txt, flags=re.M):
        defs.append(m.group(1))
    defs_by_file[f] = defs

    # all identifier mentions (rough)
    idents = set()
    for m in re.finditer(r'\b([A-Za-z_]\w*)\b', txt):
        idents.add(m.group(1))
    idents_by_file[f] = idents

# Build a global index of identifier mentions outside each file
global_mentions = {}
for f in py_files:
    txt = read_text(f)
    global_mentions[f] = txt

def is_used_elsewhere(name, this_file):
    # Check mentions in other files or in this file outside the def line
    for f in py_files:
        txt = global_mentions[f]
        # exclude definition lines in the same file
        if f == this_file:
            # remove 'def name(' and 'class name(' occurrences to avoid false positives
            txt_check = re.sub(rf'^\s*(def|class)\s+{re.escape(name)}\s*\(', '', txt, flags=re.M)
        else:
            txt_check = txt
        if re.search(rf'\b{re.escape(name)}\b', txt_check):
            return True
    return False

unused = {}
for f, defs in defs_by_file.items():
    u = []
    for name in defs:
        if not is_used_elsewhere(name, f):
            u.append(name)
    if u:
        unused[f] = u

print("=== Potentially unused definitions (approximate) ===")
for f, names in sorted(unused.items()):
    rel = os.path.relpath(f, root)
    print(f"- {rel}: {', '.join(names)}")

# Also report modules imported but not used within files (simplistic)
print("\n=== Imports that may be unused (per file, simplistic) ===")
for f in py_files:
    txt = read_text(f)
    imports = re.findall(r'^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+([A-Za-z0-9_,\s*]+)|^\s*import\s+([A-Za-z0-9_\.]+)', txt, flags=re.M)
    imported_names = set()
    for imp in imports:
        if imp[0]:  # from ... import ...
            names = [n.strip() for n in imp[1].split(",")]
            for n in names:
                imported_names.add(n)
        elif imp[2]:
            imported_names.add(imp[2].split(".")[0])
    possibly_unused = []
    for n in imported_names:
        # consider used if mentioned anywhere
        if not re.search(rf'\b{re.escape(n)}\b', txt):
            possibly_unused.append(n)
    if possibly_unused:
        rel = os.path.relpath(f, root)
        print(f"- {rel}: {', '.join(possibly_unused)}")

# Quick notes based on known structure
print("\n=== Notes ===")
print("* This is a heuristic. It flags defs not referenced by name anywhere; dynamic usage won’t be detected.")
print("* Given compute.py simplification, spectral/xrt-related functions are removed there, but ui_widgets.py still imports Attenuator and supports attenuator UI.")
print("* If attenuators are not used in compute, attenuation.transmission_curve() may be dead unless used elsewhere.")