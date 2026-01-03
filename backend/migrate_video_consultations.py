"""
Migration script to make dermatologist_id nullable in video_consultations table.
This allows consultations to be created without a pre-selected dermatologist.
"""
import sqlite3
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def migrate():
    conn = sqlite3.connect('skin_classifier.db')
    cursor = conn.cursor()

    try:
        # Clean up any leftover temp table from previous failed migration
        cursor.execute("DROP TABLE IF EXISTS video_consultations_new")

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='video_consultations'")
        if not cursor.fetchone():
            print("Table 'video_consultations' does not exist. It will be created on next backend start.")
            return

        # Get current schema
        cursor.execute("PRAGMA table_info(video_consultations)")
        columns = cursor.fetchall()

        print(f"Found {len(columns)} columns in video_consultations")

        # Check if dermatologist_id is already nullable
        derm_col = next((c for c in columns if c[1] == 'dermatologist_id'), None)
        if derm_col and not derm_col[3]:  # not_null is 0 (False)
            print("dermatologist_id is already nullable. No migration needed.")
            return

        print("Migrating table to make dermatologist_id nullable...")

        # Get column names for the SELECT statement
        column_names = [col[1] for col in columns]
        columns_str = ', '.join(column_names)

        # Build CREATE TABLE statement dynamically
        create_cols = []
        for col in columns:
            col_name = col[1]
            col_type = col[2] or 'TEXT'
            not_null = col[3]
            default_value = col[4]
            is_pk = col[5]

            col_def = f"{col_name} {col_type}"

            if is_pk:
                col_def += " PRIMARY KEY"

            # Make dermatologist_id nullable, keep others as-is
            if col_name == 'dermatologist_id':
                pass  # Don't add NOT NULL
            elif not_null and not is_pk:
                col_def += " NOT NULL"

            if default_value is not None:
                col_def += f" DEFAULT {default_value}"

            create_cols.append(col_def)

        create_sql = f"""
            CREATE TABLE video_consultations_new (
                {', '.join(create_cols)},
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (dermatologist_id) REFERENCES dermatologist_profiles(id)
            )
        """

        print("Creating new table...")
        cursor.execute(create_sql)

        # Copy data
        print("Copying data...")
        cursor.execute(f"""
            INSERT INTO video_consultations_new ({columns_str})
            SELECT {columns_str} FROM video_consultations
        """)

        # Drop old table
        print("Dropping old table...")
        cursor.execute("DROP TABLE video_consultations")

        # Rename new table
        print("Renaming new table...")
        cursor.execute("ALTER TABLE video_consultations_new RENAME TO video_consultations")

        # Recreate indexes
        print("Recreating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_video_consultations_user_id ON video_consultations(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_video_consultations_dermatologist_id ON video_consultations(dermatologist_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_video_consultations_scheduled_datetime ON video_consultations(scheduled_datetime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_video_consultations_status ON video_consultations(status)")

        conn.commit()
        print("\nMigration completed successfully!")
        print("dermatologist_id is now nullable in video_consultations table.")

    except Exception as e:
        conn.rollback()
        print(f"Error during migration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
