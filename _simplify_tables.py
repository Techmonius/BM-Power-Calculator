import io, re

path = r"C:\Users\soprondek\Dev\BM-Power-Calculator\src\ui_widgets.py"

with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def find_block(start_pattern, end_pattern):
    start = -1
    end = -1
    for i, ln in enumerate(lines):
        if start == -1 and re.search(start_pattern, ln):
            start = i
        if start != -1 and re.search(end_pattern, ln):
            end = i
            break
    return start, end

# Helper to strip style/class from th/td and table tags within a range
def strip_styles_in_range(start, end):
    for i in range(start, end+1):
        ln = lines[i]
        # table tag: remove class or style attributes
        ln = re.sub(r'<table[^>]*>', '<table>', ln)
        # th/td style attributes (single or double quotes)
        ln = re.sub(r'<th\s+style=[\'"][^\'"]*[\'"]>', '<th>', ln)
        ln = re.sub(r'<td\s+style=[\'"][^\'"]*[\'"]>', '<td>', ln)
        lines[i] = ln

# 1) Run Summary block: rows list and summary_html display
sum_start, sum_end = find_block(r"rows = \[\]", r"display\(widgets\.HTML\(summary_html\)\)")
if sum_start != -1 and sum_end != -1:
    strip_styles_in_range(sum_start, sum_end)
    # Simplify summary_html table tag to basic
    for i in range(sum_start, sum_end+1):
        if "summary_html =" in lines[i]:
            # Replace any table start with plain <table>
            lines[i] = re.sub(r'<table[^>]*>', '<table>', lines[i])
            # Also remove any class/style on the table in subsequent concatenations
        # Also ensure any occurrence of class or style in table concatenation is removed in the block
else:
    pass  # leave as-is if block not found

# 2) Per-Magnet Summary block: rows_mag and mag_html display
mag_start, mag_end = find_block(r"rows_mag = \[\]", r"display\(widgets\.HTML\(mag_html\)\)")
if mag_start != -1 and mag_end != -1:
    strip_styles_in_range(mag_start, mag_end)
    # Simplify mag_html table tag to basic
    for i in range(mag_start, mag_end+1):
        if "mag_html =" in lines[i]:
            lines[i] = re.sub(r'<table[^>]*>', '<table>', lines[i])
else:
    pass

# Write back
with io.open(path, "w", encoding="utf-8") as f:
    f.write("".join(lines))

print("Stripped styles and simplified table tags for Run Summary and Per-Magnet Summary.")