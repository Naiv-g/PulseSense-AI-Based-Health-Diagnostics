#!/usr/bin/env python3
"""
Script to create a .env file for database configuration.
"""
import os
from pathlib import Path

def create_env_file():
    env_path = Path(__file__).parent / ".env"
    
    if env_path.exists():
        print(".env file already exists!")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print("\n=== Database Configuration Setup ===\n")
    
    db_host = input("MySQL Host [localhost]: ").strip() or "localhost"
    db_user = input("MySQL User [root]: ").strip() or "root"
    db_password = input("MySQL Password (press Enter for no password): ").strip() or ""
    db_name = input("Database Name [pulsesense]: ").strip() or "pulsesense"
    db_port = input("MySQL Port [3306]: ").strip() or "3306"
    
    env_content = f"""# Database Configuration
DB_HOST={db_host}
DB_USER={db_user}
DB_PASSWORD={db_password}
DB_NAME={db_name}
DB_PORT={db_port}
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"\n✓ .env file created at {env_path}")
    print("\nPlease verify your MySQL credentials and try running the server again.")

if __name__ == "__main__":
    create_env_file()

