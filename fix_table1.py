file_path = r"c:\Users\LOQ 15IRX9\Downloads\DL_Project_2\paper_output\main_enhanced_v2.tex"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old = "R12 & Feature+temporal+refine & CNN+GRUSKIP+Trans & METR-LA & Three-stage complexity \\\\"
new = "R12 & Feature+temporal+refine & CNN+GRU- SKIP+Trans & METR-LA & Three-stage complexity \\\\"

if old in content:
    content = content.replace(old, new)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed successfully.")
else:
    # Try stripping and checking
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "R12" in line and "CNN+GRUSKIP" in line:
            print(f"Line {i+1}: {repr(line)}")
