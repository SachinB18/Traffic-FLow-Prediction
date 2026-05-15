file_path = r"c:\Users\LOQ 15IRX9\Downloads\DL_Project_2\paper_output\main_enhanced_v2.tex"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_author = r"""\author{
\IEEEauthorblockN{Sachin Bhabad}
\IEEEauthorblockA{\textit{Computer Engineering}\\
\textit{MIT Academy of Engineering, Alandi}\\
Pune, India\\
sachin.bhabad@mitaoe.ac.in}
\and
\IEEEauthorblockN{Omkar Khilare}
\IEEEauthorblockA{\textit{Computer Engineering}\\
\textit{MIT Academy of Engineering, Alandi}\\
Pune, India\\
omkar.khilare@mitaoe.ac.in}
\and
\IEEEauthorblockN{Samadhan Mane}
\IEEEauthorblockA{\textit{Computer Engineering}\\
\textit{MIT Academy of Engineering, Alandi}\\
Pune, India\\
samadhan.mane@mitaoe.ac.in}
\and
\IEEEauthorblockN{Vivek Borade}
\IEEEauthorblockA{\textit{Computer Engineering}\\
\textit{MIT Academy of Engineering, Alandi}\\
Pune, India\\
vivek.borade@mitaoe.ac.in}
}"""

new_author = r"""\author{
\IEEEauthorblockN{Sachin Bhabad, Omkar Khilare, Samadhan Mane, and Vivek Borade}
\IEEEauthorblockA{\textit{School of Computer Engineering}\\
MIT Academy of Engineering\\
Pune, India}
\and
\IEEEauthorblockN{Guide: Dr. Sunita Barve}
\IEEEauthorblockA{\textit{School of Computer Engineering}\\
MIT Academy of Engineering\\
Pune, India\\
sunita.barve@mitaoe.ac.in}
}"""

if old_author in content:
    content = content.replace(old_author, new_author)
    print("Author block updated successfully.")
else:
    print("WARNING: Could not find author block.")
    lines = content.splitlines()
    for i, l in enumerate(lines[21:50], start=22):
        print(f"{i}: {l}")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
