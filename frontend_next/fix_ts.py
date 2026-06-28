with open('src/app/page.tsx', 'r', encoding='utf-8') as f:
    c = f.read()

# Imports
import_target = "  Sparkles\n} from 'lucide-react';"
import_target_r = "  Sparkles\r\n} from 'lucide-react';"
replacement = "  Sparkles,\n  Bookmark,\n  History,\n  FileText,\n  XCircle,\n  GraduationCap,\n  Code,\n  BookOpen\n} from 'lucide-react';"

if import_target in c:
    c = c.replace(import_target, replacement)
    print("Found LF")
elif import_target_r in c:
    c = c.replace(import_target_r, replacement)
    print("Found CRLF")
else:
    print("Icon target NOT FOUND")

# Props
for p in ['onSaveJob={handleSaveJob}', 'onUnsaveJob={handleUnsaveJob}', 'isSaved={isJobSaved(job.id)}']:
    c = c.replace(p + '\n', '')
    c = c.replace(p + '\r\n', '')
    c = c.replace(p, '')

c = c.replace('onApply={handleApply}\n', 'onAutoApply={() => handleApply(job.id)}\n')
c = c.replace('onApply={handleApply}\r\n', 'onAutoApply={() => handleApply(job.id)}\r\n')
c = c.replace('onApply={handleApply}', 'onAutoApply={() => handleApply(job.id)}')

with open('src/app/page.tsx', 'w', encoding='utf-8') as f:
    f.write(c)
print("Finished patching page.tsx")
