from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
import mysql.connector
from datetime import datetime, timedelta

from .database import db
from .models import (
    User, UserCreate, UserLogin, MedicalRecord, MedicalRecordCreate,
    SymptomQuery, DiseasePrediction, Token, ChatbotQuery, ChatbotResponse
)
from .auth import (
    authenticate_user, create_access_token, get_password_hash,
    verify_token, get_user_by_username, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .rag_agent import get_rag_agent
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
    
    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check if username exists (case-insensitive)
        cursor.execute("SELECT id, username, email FROM users WHERE LOWER(username) = LOWER(%s)", 
                      (user_data.username,))
        existing_username = cursor.fetchone()
        if existing_username:
            raise HTTPException(
                status_code=400, 
                detail=f"Username '{existing_username['username']}' already registered"
            )
        
        # Check if email exists (case-insensitive)
        cursor.execute("SELECT id, username, email FROM users WHERE LOWER(email) = LOWER(%s)", 
                      (user_data.email,))
        existing_email = cursor.fetchone()
        if existing_email:
            raise HTTPException(
                status_code=400, 
                detail=f"Email '{existing_email['email']}' already registered"
            )
        
        # Create user
        hashed_password = get_password_hash(user_data.password)
        cursor.execute(
            """INSERT INTO users (username, email, full_name, date_of_birth, gender, phone, hashed_password) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (user_data.username, user_data.email, user_data.full_name, 
             user_data.date_of_birth, user_data.gender, user_data.phone, hashed_password)
        )
        
        connection.commit()
        user_id = cursor.lastrowid
        
        return {"message": "User registered successfully", "user_id": user_id}
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except mysql.connector.Error as db_error:
        error_code = db_error.errno
        error_msg = str(db_error)
        print(f"Database error during registration: {error_msg} (Error code: {error_code})")
        
        if error_code == 1062:  # Duplicate entry error
            if "username" in error_msg.lower():
                raise HTTPException(status_code=400, detail="Username already registered")
            elif "email" in error_msg.lower():
                raise HTTPException(status_code=400, detail="Email already registered")
        elif error_code == 1406:  # Data too long
            raise HTTPException(status_code=400, detail="One or more fields exceed maximum length")
        elif error_code == 1264:  # Out of range value
            raise HTTPException(status_code=400, detail="Invalid date or value format")
        else:
            raise HTTPException(status_code=500, detail=f"Database error: {error_msg}")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Unexpected error during registration: {str(e)}")
        print(f"Traceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
    finally:
        if cursor:
            cursor.close()

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

@app.delete("/medical-records/{record_id}", response_model=dict)
async def delete_medical_record(record_id: int, current_user: dict = Depends(get_current_user)):
    """Delete a medical record. Users can only delete their own records."""
    connection = db.get_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = connection.cursor()
    
    try:
        # First verify the record belongs to the user
        cursor.execute(
            "SELECT user_id FROM medical_records WHERE id = %s",
            (record_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            raise HTTPException(status_code=404, detail="Medical record not found")
        
        if result[0] != current_user["id"]:
            cursor.close()
            raise HTTPException(status_code=403, detail="You can only delete your own records")
        
        # Delete the record
        cursor.execute(
            "DELETE FROM medical_records WHERE id = %s AND user_id = %s",
            (record_id, current_user["id"])
        )
        
        connection.commit()
        cursor.close()
        
        return {"message": "Medical record deleted successfully", "record_id": record_id}
    except HTTPException:
        raise
    except Exception as e:
        cursor.close()
        raise HTTPException(status_code=500, detail=f"Failed to delete medical record: {str(e)}")

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
        rag_agent = get_rag_agent()
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


@app.get("/rag-status")
async def rag_status():
    """Return RAG agent diagnostic information so we can tell which mode is active.

    Fields:
    - embeddings: whether embeddings were created
    - vector_store: whether vector store (FAISS) was built
    - llm: whether an LLM pipeline is loaded
    - qa_chain: whether a RetrievalQA chain is available
    - mode: 'rag' if QA chain available, 'rule-based' otherwise
    """
    try:
        agent = get_rag_agent()
        status = {
            "embeddings": bool(agent.embeddings),
            "vector_store": bool(agent.vector_store),
            "llm": bool(agent.llm),
            "qa_chain": bool(agent.qa_chain),
        }
        status["mode"] = "rag" if status["qa_chain"] and status["llm"] else "rule-based"
        return status
    except Exception as e:
        return {"error": str(e)}

# Chatbot endpoint
@app.post("/chatbot/query", response_model=ChatbotResponse)
async def chatbot_query(query: ChatbotQuery, current_user: dict = Depends(get_current_user)):
    """Handle chatbot queries about diseases and store conversation in database."""
    connection = db.get_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        # Get response from RAG agent
        rag_agent = get_rag_agent()
        bot_response = rag_agent.chat_about_disease(query.message)
        
        # Store conversation in database
        cursor = connection.cursor()
        try:
            cursor.execute(
                """INSERT INTO chatbot_conversations (user_id, user_message, bot_response) 
                VALUES (%s, %s, %s)""",
                (current_user["id"], query.message, bot_response)
            )
            connection.commit()
            conversation_id = cursor.lastrowid
            cursor.close()
            
            return ChatbotResponse(response=bot_response, conversation_id=conversation_id)
        except Exception as db_error:
            cursor.close()
            # Still return the response even if database storage fails
            print(f"Failed to store conversation: {db_error}")
            return ChatbotResponse(response=bot_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot query failed: {str(e)}")

@app.get("/chatbot/history", response_model=List[dict])
async def get_chatbot_history(current_user: dict = Depends(get_current_user)):
    """Get chatbot conversation history for the current user."""
    connection = db.get_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute(
            "SELECT id, user_message, bot_response, created_at FROM chatbot_conversations WHERE user_id = %s ORDER BY created_at DESC LIMIT 50",
            (current_user["id"],)
        )
        
        conversations = cursor.fetchall()
        cursor.close()
        
        # Convert datetime to string for JSON serialization
        for conv in conversations:
            if conv["created_at"]:
                conv["created_at"] = conv["created_at"].isoformat()
        
        return conversations
    except Exception as e:
        cursor.close()
        raise HTTPException(status_code=500, detail=f"Failed to fetch chatbot history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)