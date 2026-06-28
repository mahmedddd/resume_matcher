import sqlite3
db = sqlite3.connect('pakinter.db')

def check_source(source):
    print(f"--- {source} Check ---")
    rows = db.execute("SELECT url, source, title FROM internships WHERE source LIKE ? LIMIT 5", (f'%{source}%',)).fetchall()
    if not rows:
        print("No results found.")
    for r in rows:
        print(f"[{r[1]}] url='{r[0]}' title='{r[2][:50]}'")
    print()

check_source('linkedin')
check_source('rozee')
check_source('mustakbil')
check_source('internshala')
