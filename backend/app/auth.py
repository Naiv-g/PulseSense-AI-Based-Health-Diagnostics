from jose import JWTError, jwt
from passlib.context import CryptContext # For password hashing
from datetime import datetime, timedelta
from typing import Optional
import mysql.connector
import bcrypt
from .database import db
from .models import TokenData

# Security configuration
SECRET_KEY = "your-secret-key-here"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configure CryptContext with bcrypt, using lazy initialization to avoid bug detection issues
# We'll use bcrypt directly as a fallback if passlib has issues
try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception as e:
    print(f"Warning: CryptContext initialization had issues: {e}")
    pwd_context = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    # Ensure password is a string
    if not isinstance(plain_password, str):
        plain_password = str(plain_password)
    
    # Ensure password doesn't exceed bcrypt's 72-byte limit
    password_bytes = plain_password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
        # Ensure valid UTF-8
        while password_bytes and (password_bytes[-1] & 0x80) and (password_bytes[-1] & 0xC0) != 0xC0:
            password_bytes = password_bytes[:-1]
        plain_password = password_bytes.decode('utf-8', errors='ignore')
    
    try:
        # Try using passlib first
        if pwd_context:
            return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Password verification error with passlib: {e}")
    
    # Fallback to direct bcrypt
    try:
        password_bytes = plain_password.encode('utf-8')
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
        return bcrypt.checkpw(password_bytes, hashed_password.encode('utf-8'))
    except Exception as e:
        print(f"Password verification error with bcrypt: {e}")
        return False

def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    Bcrypt has a 72-byte limit, so we ensure the password is properly encoded.
    """
    # Ensure password is a string
    if not isinstance(password, str):
        password = str(password)
    
    # Ensure password doesn't exceed bcrypt's 72-byte limit
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        # Truncate to 72 bytes (preserving valid UTF-8)
        password_bytes = password_bytes[:72]
        # Remove any incomplete UTF-8 sequences at the end
        while password_bytes and (password_bytes[-1] & 0x80) and (password_bytes[-1] & 0xC0) != 0xC0:
            password_bytes = password_bytes[:-1]
        password = password_bytes.decode('utf-8', errors='ignore')
        print(f"Warning: Password was longer than 72 bytes and was truncated")
    
    # Use direct bcrypt to avoid passlib initialization issues
    try:
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
        
        # Generate salt and hash using bcrypt directly
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    except Exception as e:
        print(f"Error using bcrypt directly: {e}")
        # Fallback to passlib if bcrypt direct fails
        if pwd_context:
            try:
                return pwd_context.hash(password)
            except Exception as e2:
                print(f"Error using passlib as fallback: {e2}")
                raise ValueError(f"Failed to hash password: {e2}")
        raise ValueError(f"Failed to hash password: {e}")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[TokenData]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return TokenData(username=username)
    except JWTError:
        return None

def get_user_by_username(username: str):
    connection = db.get_connection()
    cursor = connection.cursor(dictionary=True)
    
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    user = cursor.fetchone()
    
    cursor.close()
    return user

def authenticate_user(username: str, password: str):
    user = get_user_by_username(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user