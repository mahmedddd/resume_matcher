import sqlite3

db_path = "pakinter.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("PRAGMA table_info(user_profile);")
    columns = cursor.fetchall()
    print("Columns in 'user_profile':")
    for col in columns:
        print(f" - {col[1]} ({col[2]})")

    cursor.execute("PRAGMA table_info(applications);")
    cols2 = cursor.fetchall()
    print("\nColumns in 'applications':")
    for col in cols2:
        print(f" - {col[1]}")

except Exception as e:
    print(f"Error reading schema: {e}")
finally:
    conn.close()
