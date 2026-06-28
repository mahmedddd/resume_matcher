import sqlite3
db = sqlite3.connect('pakinter.db')
# Check Rozee specifically
rows = db.execute("SELECT url, source, title FROM internships WHERE source = 'rozee' LIMIT 10").fetchall()
print("--- Rozee Check ---")
for r in rows:
    print(f"[{r[1]}] url='{r[0]}' title='{r[2][:50]}'")

# Check Internshala structure again
rows = db.execute("SELECT url, source, title FROM internships WHERE source = 'internshala' LIMIT 5").fetchall()
print("\n--- Internshala Check ---")
for r in rows:
    print(f"[{r[1]}] url='{r[0]}' title='{r[2][:50]}'")
