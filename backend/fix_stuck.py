import sqlite3

msg = "Agent process died - backend was restarted. Please re-apply."
conn = sqlite3.connect("pakinter.db")
c = conn.cursor()
c.execute(
    "UPDATE applications SET status=?, notes=? WHERE status=?",
    ("Failed", msg, "Applying"),
)
conn.commit()
print(f"Fixed {c.rowcount} stuck applications")
conn.close()
