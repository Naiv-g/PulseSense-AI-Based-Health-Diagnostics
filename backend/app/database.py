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
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                # Additional options for remote connections
                connect_timeout=30,
                buffered=True
            )
            print(f"Connected to MySQL database: {self.database} on {self.host}:{self.port}")
            return self.connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None
    
    def get_connection(self):
        if self.connection is None or not self.connection.is_connected():
            return self.connect()
        return self.connection

db = Database()