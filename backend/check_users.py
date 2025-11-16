#!/usr/bin/env python3
"""
Script to check existing users in the database.
"""
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import db

def check_existing_users():
    """Check all existing users in the database."""
    connection = db.get_connection()
    if not connection:
        print("❌ Failed to connect to database")
        return
    
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Get all users
        cursor.execute("SELECT id, username, email, full_name, created_at FROM users ORDER BY id")
        users = cursor.fetchall()
        
        if not users:
            print("✅ No users found in the database.")
            return
        
        print(f"\n📊 Found {len(users)} user(s) in the database:\n")
        print("=" * 80)
        print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Full Name':<20}")
        print("=" * 80)
        
        for user in users:
            print(f"{user['id']:<5} {user['username']:<20} {user['email']:<30} {user['full_name']:<20}")
        
        print("=" * 80)
        
        # Check for specific username/email
        print("\n🔍 Checking for potential conflicts...")
        test_username = "Adiraw101"
        test_email = "adiraw101@gmail.com"
        
        # Case-insensitive check
        cursor.execute("SELECT id, username, email FROM users WHERE LOWER(username) = LOWER(%s)", 
                      (test_username,))
        existing_username = cursor.fetchone()
        
        cursor.execute("SELECT id, username, email FROM users WHERE LOWER(email) = LOWER(%s)", 
                      (test_email,))
        existing_email = cursor.fetchone()
        
        if existing_username:
            print(f"\n⚠️  Username conflict found:")
            print(f"   Database has: '{existing_username['username']}' (case-insensitive match with '{test_username}')")
        
        if existing_email:
            print(f"\n⚠️  Email conflict found:")
            print(f"   Database has: '{existing_email['email']}' (case-insensitive match with '{test_email}')")
        
        if not existing_username and not existing_email:
            print(f"\n✅ No conflicts found for username '{test_username}' or email '{test_email}'")
            print("   The registration should work. If it doesn't, check backend logs.")
        
    except Exception as e:
        print(f"❌ Error checking users: {e}")
    finally:
        cursor.close()

if __name__ == "__main__":
    print("🔍 Checking existing users in database...\n")
    check_existing_users()

