import sqlite3

db_path = "pakinter.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    print("Adding 'experience' column...")
    cursor.execute("ALTER TABLE user_profile ADD COLUMN experience JSON;")
except Exception as e:
    print(f"Skipping experience: {e}")

try:
    print("Adding 'projects' column...")
    cursor.execute("ALTER TABLE user_profile ADD COLUMN projects JSON;")
except Exception as e:
    print(f"Skipping projects: {e}")

conn.commit()
conn.close()
print("Migration applied successfully.")
