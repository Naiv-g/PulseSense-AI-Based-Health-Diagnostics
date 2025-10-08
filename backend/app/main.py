from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
import mysql.connector
from datetime import datetime, timedelta

from .database import db
from .models import (
    User, UserCreate, UserLogin, MedicalRecord, MedicalRecordCreate,
    SymptomQuery, DiseasePrediction, Token
)
from .auth import (
    authenticate_user, create_access_token, get_password_hash,
    verify_token, get_user_by_username, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .rag_agent import rag_agent
from .schemas import CREATE_TABLES, SAMPLE_SYMPTOMS, SAMPLE_DISEASES

app = FastAPI(title="PulseSense API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    print("Starting PulseSense backend...")
    connection = db.connect()
    if connection:
        cursor = connection.cursor()
        
        # Create tables
        for statement in CREATE_TABLES.split(';'):
            if statement.strip():
                try:
                    cursor.execute(statement)
                except mysql.connector.Error as e:
                    print(f"Error creating table: {e}")
        
        # Insert sample symptoms
        for symptom in SAMPLE_SYMPTOMS:
            try:
                cursor.execute(
                    "INSERT IGNORE INTO symptoms (name, category) VALUES (%s, %s)",
                    (symptom, "General")
                )
            except mysql.connector.Error:
                pass
        
        # Insert sample diseases
        for disease_data in SAMPLE_DISEASES:
            try:
                cursor.execute(
                    "INSERT IGNORE INTO diseases (name, description, specialist) VALUES (%s, %s, %s)",
                    (disease_data["name"], disease_data["description"], disease_data["specialist"])
                )
                
                # Get disease ID
                cursor.execute("SELECT id FROM diseases WHERE name = %s", (disease_data["name"],))
                result = cursor.fetchone()
                if result:
                    disease_id = result[0]
                    
                    # Link symptoms
                    for symptom_name in disease_data["symptoms"]:
                        cursor.execute("SELECT id FROM symptoms WHERE name = %s", (symptom_name,))
                        symptom_result = cursor.fetchone()
                        if symptom_result:
                            symptom_id = symptom_result[0]
                            cursor.execute(
                                "INSERT IGNORE INTO disease_symptoms (disease_id, symptom_id) VALUES (%s, %s)",
                                (disease_id, symptom_id)
                            )
            except mysql.connector.Error as e:
                print(f"Error inserting disease: {e}")
        
        connection.commit()
        cursor.close()
        print("Database initialized successfully")
    else:
        print("Failed to connect to database")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    token_data = verify_token(token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_username(token_data.username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Authentication endpoints
@app.post("/register", response_model=dict)
async def register(user_data: UserCreate):
    connection = db.get_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
        
    cursor = connection.cursor(dictionary=True)
    
    # Check if user exists
    cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", 
                  (user_data.username, user_data.email))
    if cursor.fetchone():
        cursor.close()
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    try:
        cursor.execute(
            """INSERT INTO users (username, email, full_name, date_of_birth, gender, phone, hashed_password) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (user_data.username, user_data.email, user_data.full_name, 
             user_data.date_of_birth, user_data.gender, user_data.phone, hashed_password)
        )
        
        connection.commit()
        user_id = cursor.lastrowid
        cursor.close()
        
        return {"message": "User registered successfully", "user_id": user_id}
    except Exception as e:
        cursor.close()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    user = authenticate_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Medical records endpoints
@app.post("/medical-records", response_model=dict)
async def create_medical_record(record: MedicalRecordCreate, current_user: dict = Depends(get_current_user)):
    connection = db.get_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
        
    cursor = connection.cursor()
    
    try:
        cursor.execute(
            """INSERT INTO medical_records (user_id, record_type, description, diagnosis, treatment, record_date) 
            VALUES (%s, %s, %s, %s, %s, %s)""",
            (current_user["id"], record.record_type, record.description, 
             record.diagnosis, record.treatment, record.record_date)
        )
        
        connection.commit()
        record_id = cursor.lastrowid
        cursor.close()
        
        return {"message": "Medical record created successfully", "record_id": record_id}
    except Exception as e:
        cursor.close()
        raise HTTPException(status_code=500, detail=f"Failed to create medical record: {str(e)}")

@app.get("/medical-records", response_model=List[dict])
async def get_medical_records(current_user: dict = Depends(get_current_user)):
    connection = db.get_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
        
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute(
            "SELECT * FROM medical_records WHERE user_id = %s ORDER BY record_date DESC",
            (current_user["id"],)
        )
        
        records = cursor.fetchall()
        cursor.close()
        
        # Convert dates to string for JSON serialization
        for record in records:
            if record["record_date"]:
                record["record_date"] = record["record_date"].isoformat()
            if record["created_at"]:
                record["created_at"] = record["created_at"].isoformat()
        
        return records
    except Exception as e:
        cursor.close()
        raise HTTPException(status_code=500, detail=f"Failed to fetch medical records: {str(e)}")

# Symptoms and disease prediction endpoints
@app.get("/symptoms", response_model=List[str])
async def get_symptoms():
    connection = db.get_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
        
    cursor = connection.cursor()
    
    try:
        cursor.execute("SELECT name FROM symptoms ORDER BY name")
        symptoms = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        return symptoms
    except Exception as e:
        cursor.close()
        raise HTTPException(status_code=500, detail=f"Failed to fetch symptoms: {str(e)}")

@app.post("/predict-disease", response_model=DiseasePrediction)
async def predict_disease(query: SymptomQuery, current_user: dict = Depends(get_current_user)):
    try:
        prediction = rag_agent.predict_disease(query.symptoms)
        
        # Store the query in database (optional)
        connection = db.get_connection()
        if connection:
            cursor = connection.cursor()
            try:
                cursor.execute(
                    """INSERT INTO medical_records (user_id, record_type, description, record_date) 
                    VALUES (%s, %s, %s, %s)""",
                    (current_user["id"], "Symptom Check", 
                     f"Symptom query: {', '.join(query.symptoms)} - Prediction: {prediction['disease']}",
                     datetime.now().date())
                )
                connection.commit()
                cursor.close()
            except:
                if cursor:
                    cursor.close()
        
        return DiseasePrediction(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# User profile endpoint
@app.get("/profile", response_model=dict)
async def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "date_of_birth": current_user["date_of_birth"].isoformat() if current_user["date_of_birth"] else "",
        "gender": current_user["gender"],
        "phone": current_user["phone"]
    }

@app.get("/")
async def root():
    return {"message": "PulseSense API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)