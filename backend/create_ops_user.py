"""
Create an Operations Staff user for the platform.
This user can access the Operations Dashboard to manage consultation assignments.
"""
import sqlite3
import os
from passlib.context import CryptContext

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_ops_user():
    conn = sqlite3.connect('skin_classifier.db')
    cursor = conn.cursor()

    try:
        # Check if ops user already exists
        cursor.execute("SELECT id, username, role FROM users WHERE username = 'ops_admin'")
        existing = cursor.fetchone()

        if existing:
            print(f"Ops user already exists: ID={existing[0]}, username={existing[1]}, role={existing[2]}")
            # Update role if needed
            if existing[2] != 'ops_staff':
                cursor.execute("UPDATE users SET role = 'ops_staff' WHERE username = 'ops_admin'")
                conn.commit()
                print("Updated role to 'ops_staff'")
            return

        # Create ops user
        password = "OpsAdmin123!"
        hashed_password = pwd_context.hash(password)

        cursor.execute("""
            INSERT INTO users (
                username, email, full_name, hashed_password, is_active, role, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            'ops_admin',
            'ops@dermaintel.com',
            'Operations Admin',
            hashed_password,
            1,
            'ops_staff'
        ))

        conn.commit()
        user_id = cursor.lastrowid

        print("=" * 50)
        print("Operations Staff User Created Successfully!")
        print("=" * 50)
        print(f"Username: ops_admin")
        print(f"Password: {password}")
        print(f"Role: ops_staff")
        print(f"User ID: {user_id}")
        print("=" * 50)
        print("\nThis user can access the Operations Dashboard to:")
        print("  - View all pending consultation requests")
        print("  - Assign dermatologists to consultations")
        print("  - Monitor platform consultation activity")
        print("=" * 50)

    except Exception as e:
        conn.rollback()
        print(f"Error creating ops user: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    create_ops_user()
