from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class User(BaseModel):
    id: Optional[int] = None
    username: str
    email: str
    full_name: str
    date_of_birth: str
    gender: str
    phone: str
    hashed_password: str
    created_at: Optional[datetime] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    date_of_birth: str
    gender: str
    phone: str

class UserLogin(BaseModel):
    username: str
    password: str

class MedicalRecord(BaseModel):
    id: Optional[int] = None
    user_id: int
    record_type: str
    description: str
    diagnosis: Optional[str] = None
    treatment: Optional[str] = None
    record_date: str
    created_at: Optional[datetime] = None

class MedicalRecordCreate(BaseModel):
    record_type: str
    description: str
    diagnosis: Optional[str] = None
    treatment: Optional[str] = None
    record_date: str

class SymptomQuery(BaseModel):
    symptoms: List[str]
    user_id: int

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float
    description: str
    recommended_specialist: str
    suggested_tests: List[str]
    precautions: List[str]

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None