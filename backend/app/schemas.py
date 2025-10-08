from pydantic import BaseModel
from typing import List

# Database table schemas
CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender ENUM('Male', 'Female', 'Other') NOT NULL,
    phone VARCHAR(15) NOT NULL,
    hashed_password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS medical_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    record_type ENUM('Diagnosis', 'Treatment', 'Surgery', 'Medication', 'Allergy', 'Chronic Condition', 'Other') NOT NULL,
    description TEXT NOT NULL,
    diagnosis VARCHAR(200),
    treatment TEXT,
    record_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS symptoms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS diseases (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    specialist VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS disease_symptoms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    disease_id INT,
    symptom_id INT,
    FOREIGN KEY (disease_id) REFERENCES diseases(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
"""

# Sample data
SAMPLE_SYMPTOMS = [
    "fever", "headache", "cough", "fatigue", "nausea", "vomiting", "diarrhea",
    "chest pain", "shortness of breath", "dizziness", "rash", "abdominal pain",
    "muscle pain", "sore throat", "runny nose", "sneezing", "body aches",
    "chills", "sweating", "loss of appetite", "weight loss", "weight gain",
    "insomnia", "anxiety", "depression", "joint pain", "back pain"
]

SAMPLE_DISEASES = [
    {
        "name": "Common Cold",
        "description": "Viral infection of the upper respiratory tract",
        "specialist": "General Physician",
        "symptoms": ["cough", "sore throat", "runny nose", "sneezing", "headache"]
    },
    {
        "name": "Influenza",
        "description": "Viral infection affecting respiratory system",
        "specialist": "General Physician",
        "symptoms": ["fever", "cough", "body aches", "headache", "fatigue", "chills"]
    },
    {
        "name": "Migraine",
        "description": "Neurological condition characterized by intense headaches",
        "specialist": "Neurologist",
        "symptoms": ["headache", "nausea", "dizziness", "sensitivity to light"]
    },
    {
        "name": "Gastroenteritis",
        "description": "Inflammation of the stomach and intestines",
        "specialist": "Gastroenterologist",
        "symptoms": ["nausea", "vomiting", "diarrhea", "abdominal pain", "fever"]
    }
]