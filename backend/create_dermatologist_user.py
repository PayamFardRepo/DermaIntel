"""
Create a Dermatologist user (Dr. Smith) for the demo.
This user can view assigned consultations and access clinical tools.
"""
import sqlite3
import os
from passlib.context import CryptContext

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_dermatologist():
    conn = sqlite3.connect('skin_classifier.db')
    cursor = conn.cursor()

    try:
        # Check if dr_smith user already exists
        cursor.execute("SELECT id, username, role FROM users WHERE username = 'dr_smith'")
        existing = cursor.fetchone()

        if existing:
            print(f"Dr. Smith user already exists: ID={existing[0]}, username={existing[1]}, role={existing[2]}")
            user_id = existing[0]
            # Update role if needed
            if existing[2] != 'dermatologist':
                cursor.execute("UPDATE users SET role = 'dermatologist', full_name = 'Dr. James Smith' WHERE id = ?", (user_id,))
                conn.commit()
                print("Updated role to 'dermatologist'")
        else:
            # Create dr_smith user
            password = "123456"
            hashed_password = pwd_context.hash(password)

            cursor.execute("""
                INSERT INTO users (
                    username, email, full_name, hashed_password, is_active, role,
                    account_type, display_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                'dr_smith',
                'dr.smith@dermaintel.com',
                'Dr. James Smith',
                hashed_password,
                1,
                'dermatologist',
                'professional',
                'professional'
            ))

            conn.commit()
            user_id = cursor.lastrowid
            print(f"Created user dr_smith with ID={user_id}")

        # Check if dermatologist profile exists
        cursor.execute("SELECT id FROM dermatologist_profiles WHERE full_name LIKE '%Smith%' OR user_id = ?", (user_id,))
        existing_profile = cursor.fetchone()

        if existing_profile:
            print(f"Dermatologist profile already exists: ID={existing_profile[0]}")
            derm_id = existing_profile[0]
        else:
            # Create dermatologist profile with correct column names
            cursor.execute("""
                INSERT INTO dermatologist_profiles (
                    user_id, full_name, credentials, email, phone_number,
                    specializations, practice_name, practice_address,
                    city, state, country, zip_code,
                    accepts_video_consultations, accepts_referrals, accepts_second_opinions,
                    video_platform, average_rating, total_reviews,
                    years_experience, medical_school, residency,
                    board_certifications, is_active, is_verified, timezone,
                    bio, availability_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                user_id,
                'Dr. James Smith',
                'MD, FAAD',
                'dr.smith@dermaintel.com',
                '(415) 555-0123',
                '["General Dermatology", "Skin Cancer", "Mohs Surgery"]',
                'DermaIntel Medical Center',
                '123 Medical Plaza, Suite 100',
                'San Francisco',
                'CA',
                'USA',
                '94102',
                1,  # accepts_video_consultations
                1,  # accepts_referrals
                1,  # accepts_second_opinions
                'zoom',
                4.8,  # average_rating
                127,  # total_reviews
                15,   # years_experience
                'Stanford University School of Medicine',
                'UCSF Dermatology Residency',
                '["American Board of Dermatology", "Mohs Micrographic Surgery"]',
                1,  # is_active
                1,  # is_verified
                'America/Los_Angeles',
                'Board-certified dermatologist specializing in skin cancer detection and treatment. Over 15 years of experience in general dermatology and Mohs surgery.',
                'available'
            ))

            conn.commit()
            derm_id = cursor.lastrowid
            print(f"Created dermatologist profile with ID={derm_id}")

        print("\n" + "=" * 50)
        print("Dermatologist Account Ready!")
        print("=" * 50)
        print(f"Username: dr_smith")
        print(f"Password: 123456")
        print(f"Role: dermatologist")
        print(f"Name: Dr. James Smith, MD, FAAD")
        print(f"Dermatologist Profile ID: {derm_id}")
        print("=" * 50)
        print("\nThis user can:")
        print("  - View assigned consultations")
        print("  - Access clinical tools and staging")
        print("  - Use professional analytics")
        print("=" * 50)

    except Exception as e:
        conn.rollback()
        print(f"Error creating dermatologist: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    create_dermatologist()
