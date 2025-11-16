import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_NAME", "pulsesense")
        self.port = os.getenv("DB_PORT", "3306")
        self.connection = None
    
    def connect(self):
        try:
            # Try connecting with provided password
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password if self.password else None,
                database=self.database,
                port=int(self.port) if self.port else 3306,
                # Additional options for remote connections
                connect_timeout=30,
                buffered=True
            )
            print(f"Connected to MySQL database: {self.database} on {self.host}:{self.port}")
            return self.connection
        except Error as e:
            error_msg = str(e)
            print(f"Error connecting to MySQL: {error_msg}")
            
            # Provide helpful error messages
            if "Access denied" in error_msg:
                print("\n" + "="*60)
                print("DATABASE CONNECTION ERROR - Access Denied")
                print("="*60)
                print("Possible solutions:")
                print("1. Create a .env file in the backend/ directory with:")
                print("   DB_HOST=localhost")
                print("   DB_USER=root")
                print("   DB_PASSWORD=your_mysql_password")
                print("   DB_NAME=pulsesense")
                print("   DB_PORT=3306")
                print("\n2. Verify your MySQL root password is correct")
                print("3. If MySQL has no password, leave DB_PASSWORD empty:")
                print("   DB_PASSWORD=")
                print("\n4. Ensure MySQL server is running")
                print("="*60 + "\n")
            elif "Can't connect" in error_msg or "Unknown MySQL server host" in error_msg:
                print("\n" + "="*60)
                print("DATABASE CONNECTION ERROR - Cannot Reach Server")
                print("="*60)
                print("Please ensure MySQL server is running and accessible.")
                print("="*60 + "\n")
            
            return None
    
    def get_connection(self):
        if self.connection is None or not self.connection.is_connected():
            return self.connect()
        return self.connection

db = Database()