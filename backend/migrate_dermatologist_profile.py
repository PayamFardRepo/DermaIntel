"""
Add missing columns to dermatologist_profiles table.
"""
import sqlite3
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def migrate():
    conn = sqlite3.connect('skin_classifier.db')
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute('PRAGMA table_info(dermatologist_profiles)')
    existing_columns = {row[1] for row in cursor.fetchall()}
    print(f"Existing columns: {existing_columns}")

    # Columns to add
    columns_to_add = [
        ("latitude", "FLOAT"),
        ("longitude", "FLOAT"),
        ("current_queue_size", "INTEGER DEFAULT 0"),
        ("max_queue_size", "INTEGER DEFAULT 10"),
        ("avg_response_hours", "FLOAT DEFAULT 24.0"),
    ]

    for col_name, col_type in columns_to_add:
        if col_name not in existing_columns:
            try:
                sql = f"ALTER TABLE dermatologist_profiles ADD COLUMN {col_name} {col_type}"
                cursor.execute(sql)
                print(f"Added column: {col_name}")
            except Exception as e:
                print(f"Error adding {col_name}: {e}")
        else:
            print(f"Column already exists: {col_name}")

    conn.commit()
    conn.close()
    print("\nMigration complete!")

if __name__ == "__main__":
    migrate()
