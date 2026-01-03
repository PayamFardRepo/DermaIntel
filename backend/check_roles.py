import sqlite3
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
conn = sqlite3.connect('skin_classifier.db')
cursor = conn.cursor()

# Check dermatologist_profiles columns
print("dermatologist_profiles columns:")
cursor.execute('PRAGMA table_info(dermatologist_profiles)')
cols = cursor.fetchall()
for c in cols:
    print(f'  {c[1]}: {c[2]}')

# Check all users and their roles
print("\nAll users and roles:")
cursor.execute("SELECT id, username, full_name, email, role FROM users ORDER BY id")
users = cursor.fetchall()
for u in users:
    print(f"  ID={u[0]}, username={u[1]}, name={u[2]}, email={u[3]}, role={u[4]}")

# Check dermatologist profiles
print("\nDermatologist profiles:")
cursor.execute("SELECT id, full_name, credentials, specializations, is_active FROM dermatologist_profiles LIMIT 10")
derms = cursor.fetchall()
for d in derms:
    print(f"  ID={d[0]}, name={d[1]}, credentials={d[2]}, specs={d[3]}, active={d[4]}")

# Check if any user has dermatologist role
print("\nUsers with dermatologist role:")
cursor.execute("SELECT id, username, full_name FROM users WHERE role = 'dermatologist'")
derm_users = cursor.fetchall()
if derm_users:
    for u in derm_users:
        print(f"  ID={u[0]}, username={u[1]}, name={u[2]}")
else:
    print("  None found!")

conn.close()
