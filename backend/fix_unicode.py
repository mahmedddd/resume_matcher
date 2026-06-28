with open('automation/form_filler.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    '\u2192': '->',
    '\u2014': '-',
    '\u2500': '-',
    '\u2502': '|',
    '\u2019': "'",
    '\u201c': '"',
    '\u201d': '"',
}

for char, replacement in replacements.items():
    content = content.replace(char, replacement)

non_ascii = [(i+1, line) for i, line in enumerate(content.splitlines()) if any(ord(c) > 127 for c in line)]
if non_ascii:
    print('Still has non-ascii:')
    for ln, line in non_ascii:
        print(f'  Line {ln}: {repr(line)}')
else:
    with open('automation/form_filler.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Done - all non-ASCII chars replaced.')
