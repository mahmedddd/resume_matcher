import sqlite3
db = sqlite3.connect('pakinter.db')
cursor = db.cursor()

# Count before
cursor.execute("SELECT COUNT(*) FROM internships WHERE url = '' OR url IS NULL")
count = cursor.fetchone()[0]
print(f"Found {count} internships with empty URLs.")

# Delete
cursor.execute("DELETE FROM internships WHERE url = '' OR url IS NULL")
db.commit()

print(f"Deleted {count} broken entries.")
db.close()
