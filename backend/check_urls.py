import sqlite3
db = sqlite3.connect('pakinter.db')
# Check all sources
rows = db.execute("SELECT url, source, title FROM internships ORDER BY scraped_at DESC LIMIT 15").fetchall()
for r in rows:
    print(f"[{r[1]}] url='{r[0]}' title='{r[2][:50]}'")
