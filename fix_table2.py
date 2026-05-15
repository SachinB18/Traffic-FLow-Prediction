file_path = r"c:\Users\LOQ 15IRX9\Downloads\DL_Project_2\paper_output\main_enhanced_v2.tex"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old = r"\begin{tabular}{@{}ll@{}}"+ "\r\n" + r"\toprule"+ "\r\n" + r"\textbf{Property} & \textbf{Value} \\"
new = r"\begin{tabular}{@{}p{3.2cm}p{4.4cm}@{}}"+ "\r\n" + r"\toprule"+ "\r\n" + r"\textbf{Property} & \textbf{Value} \\"

if old in content:
    content = content.replace(old, new)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed successfully.")
else:
    # Diagnose
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "tabular" in line and "ll" in line:
            print(f"Line {i+1}: {repr(line)}")
