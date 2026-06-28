"""
Precise line-based patch for page.tsx:
Converts Work Geometry (line 592 area) and Structural Projects (line 613 area)
from truncated slice(0,2) lists to scrollable full containers.
"""
import re

with open('src/app/page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# --- Patch 1: Work Geometry slice(0,2) → all items in scrollable container ---
# Target: the entire Work Geometry block
WG_OLD = re.compile(
    r'(\s*)\{matchingResults\.cv_skills\.work_experience\?\.length > 0 && \(\n'
    r'\s*<div className="bg-\[#0a0f1c\]/50 p-5 rounded-2xl border border-white/5">\n'
    r'\s*<h4 className="text-xs font-black tracking-widest text-slate-500 uppercase flex items-center gap-2 mb-4">\n'
    r'\s*<Briefcase size=\{14\} className="text-indigo-400"/> Work Geometry\n'
    r'\s*</h4>\n'
    r'\s*<div className="flex flex-col gap-4 text-sm text-slate-300">\n'
    r'\s*\{matchingResults\.cv_skills\.work_experience\.slice\(0,2\)\.map\(\(exp: any, i: number\) => \(\n'
    r'\s*<div key=\{i\} className="pl-3 border-l-2 border-indigo-500/30">\n'
    r'\s*<div className="font-bold text-indigo-300">\{exp\.title\} <span className="text-slate-500 font-medium">@ \{exp\.company\}</span></div>\n'
    r'\s*<div className="text-\[0\.7rem\] text-slate-500 mb-1">\{exp\.duration\}</div>\n'
    r'\s*<div className="text-xs opacity-80 line-clamp-2">\{exp\.achievements\?\.\[0\]\}</div>\n'
    r'\s*</div>\n'
    r'\s*\)\)\}\n'
    r'\s*</div>\n'
    r'\s*</div>\n'
    r'\s*\)\}',
    re.MULTILINE
)

def wg_replacement(m):
    ind = '                          '
    i2 = '                            '
    i3 = '                              '
    i4 = '                                '
    i5 = '                                  '
    return (
        f'{ind}{{matchingResults.cv_skills.work_experience?.length > 0 && (\n'
        f'{i2}<div className="bg-[#0a0f1c]/50 p-5 rounded-2xl border border-white/5 flex flex-col">\n'
        f'{i3}<h4 className="text-xs font-black tracking-widest text-slate-500 uppercase flex items-center gap-2 mb-3 shrink-0">\n'
        f'{i3}  <Briefcase size={{14}} className="text-indigo-400"/> Work Geometry\n'
        f'{i3}</h4>\n'
        f'{i3}<div className="flex flex-col gap-3 text-sm text-slate-300 overflow-y-auto max-h-52 pr-1 custom-scrollbar">\n'
        f'{i4}{{matchingResults.cv_skills.work_experience.map((exp: any, i: number) => (\n'
        f'{i5}<div key={{i}} className="pl-3 border-l-2 border-indigo-500/30 py-1">\n'
        f'{i5}  <div className="font-bold text-indigo-300">{{exp.title}} <span className="text-slate-500 font-medium">@ {{exp.company}}</span></div>\n'
        f'{i5}  <div className="text-[0.7rem] text-slate-500 mb-1">{{exp.duration}}</div>\n'
        f'{i5}  {{exp.achievements?.slice(0,2).map((ach: string, j: number) => (\n'
        f'{i5}    <div key={{j}} className="text-xs opacity-75 flex gap-1.5"><span className="text-indigo-500 shrink-0">&#8250;</span> {{ach}}</div>\n'
        f'{i5}  ))}}\n'
        f'{i5}</div>\n'
        f'{i4}))}}\n'
        f'{i3}</div>\n'
        f'{i2}</div>\n'
        f'{ind})}}' 
    )

# --- Patch 2: Structural Projects slice(0,2) → all items in scrollable container ---
SP_OLD = re.compile(
    r'(\s*)\{matchingResults\.cv_skills\.projects\?\.length > 0 && \(\n'
    r'\s*<div className="bg-\[#0a0f1c\]/50 p-5 rounded-2xl border border-white/5">\n'
    r'\s*<h4 className="text-xs font-black tracking-widest text-slate-500 uppercase flex items-center gap-2 mb-4">\n'
    r'\s*<Sparkles size=\{14\} className="text-emerald-400"/> Structural Projects\n'
    r'\s*</h4>\n'
    r'\s*<div className="flex flex-col gap-4 text-sm text-slate-300">\n'
    r'\s*\{matchingResults\.cv_skills\.projects\.slice\(0,2\)\.map\(\(proj: any, i: number\) => \(\n'
    r'\s*<div key=\{i\} className="pl-3 border-l-2 border-emerald-500/30">\n'
    r'\s*<div className="font-bold text-emerald-300">\{proj\.name\}</div>\n'
    r'\s*<div className="text-\[0\.7rem\] text-slate-500 mb-1">\{proj\.technologies\?\.join\(", "\)\}</div>\n'
    r'\s*<div className="text-xs opacity-80 line-clamp-2">\{proj\.description\}</div>\n'
    r'\s*</div>\n'
    r'\s*\)\)\}\n'
    r'\s*</div>\n'
    r'\s*</div>\n'
    r'\s*\)\}',
    re.MULTILINE
)

def sp_replacement(m):
    ind = '                          '
    i2 = '                            '
    i3 = '                              '
    i4 = '                                '
    i5 = '                                  '
    return (
        f'{ind}{{matchingResults.cv_skills.projects?.length > 0 && (\n'
        f'{i2}<div className="bg-[#0a0f1c]/50 p-5 rounded-2xl border border-white/5 flex flex-col">\n'
        f'{i3}<h4 className="text-xs font-black tracking-widest text-slate-500 uppercase flex items-center gap-2 mb-3 shrink-0">\n'
        f'{i3}  <Sparkles size={{14}} className="text-emerald-400"/> Structural Projects\n'
        f'{i3}</h4>\n'
        f'{i3}<div className="flex flex-col gap-3 text-sm text-slate-300 overflow-y-auto max-h-52 pr-1 custom-scrollbar">\n'
        f'{i4}{{matchingResults.cv_skills.projects.map((proj: any, i: number) => (\n'
        f'{i5}<div key={{i}} className="pl-3 border-l-2 border-emerald-500/30 py-1">\n'
        f'{i5}  <div className="font-bold text-emerald-300">{{proj.name}}</div>\n'
        f'{i5}  <div className="text-[0.7rem] text-slate-500 mb-1">{{proj.technologies?.join(", ")}}</div>\n'
        f'{i5}  <div className="text-xs opacity-75">{{proj.description}}</div>\n'
        f'{i5}</div>\n'
        f'{i4}))}}\n'
        f'{i3}</div>\n'
        f'{i2}</div>\n'
        f'{ind})}}' 
    )

new_content, n1 = WG_OLD.subn(wg_replacement, content)
new_content, n2 = SP_OLD.subn(sp_replacement, new_content)

print(f'Work Geometry replacements: {n1}')
print(f'Structural Projects replacements: {n2}')

if n1 + n2 > 0:
    with open('src/app/page.tsx', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('✅ page.tsx saved successfully.')
else:
    print('❌ No patterns matched. Check the regex.')
