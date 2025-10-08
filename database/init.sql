-- Create database
CREATE DATABASE IF NOT EXISTS pulsesense;
USE pulsesense;

-- Users table
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

-- Medical records table
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

-- Symptoms table
CREATE TABLE IF NOT EXISTS symptoms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50)
);

-- Diseases table
CREATE TABLE IF NOT EXISTS diseases (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    specialist VARCHAR(50)
);

-- Disease-symptoms mapping table
CREATE TABLE IF NOT EXISTS disease_symptoms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    disease_id INT,
    symptom_id INT,
    FOREIGN KEY (disease_id) REFERENCES diseases(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);

-- Insert sample symptoms
INSERT IGNORE INTO symptoms (name, category) VALUES
('fever', 'General'),
('headache', 'Neurological'),
('cough', 'Respiratory'),
('fatigue', 'General'),
('nausea', 'Digestive'),
('vomiting', 'Digestive'),
('diarrhea', 'Digestive'),
('chest pain', 'Cardiac'),
('shortness of breath', 'Respiratory'),
('dizziness', 'Neurological'),
('rash', 'Dermatological'),
('abdominal pain', 'Digestive'),
('muscle pain', 'Musculoskeletal'),
('sore throat', 'Respiratory'),
('runny nose', 'Respiratory'),
('sneezing', 'Respiratory'),
('body aches', 'Musculoskeletal'),
('chills', 'General'),
('sweating', 'General'),
('loss of appetite', 'General'),
('weight loss', 'General'),
('weight gain', 'General'),
('insomnia', 'Neurological'),
('anxiety', 'Psychological'),
('depression', 'Psychological'),
('joint pain', 'Musculoskeletal'),
('back pain', 'Musculoskeletal');

-- Insert sample diseases
INSERT IGNORE INTO diseases (name, description, specialist) VALUES
('Common Cold', 'Viral infection of the upper respiratory tract', 'General Physician'),
('Influenza', 'Viral infection affecting respiratory system', 'General Physician'),
('Migraine', 'Neurological condition characterized by intense headaches', 'Neurologist'),
('Gastroenteritis', 'Inflammation of the stomach and intestines', 'Gastroenterologist'),
('Hypertension', 'High blood pressure condition', 'Cardiologist'),
('Diabetes', 'Metabolic disorder characterized by high blood sugar', 'Endocrinologist'),
('Asthma', 'Chronic inflammatory disease of the airways', 'Pulmonologist'),
('Arthritis', 'Inflammation of one or more joints', 'Rheumatologist');

-- Link diseases with symptoms
INSERT IGNORE INTO disease_symptoms (disease_id, symptom_id)
SELECT d.id, s.id FROM diseases d
JOIN symptoms s ON s.name IN ('cough', 'sore throat', 'runny nose', 'sneezing', 'headache')
WHERE d.name = 'Common Cold';

INSERT IGNORE INTO disease_symptoms (disease_id, symptom_id)
SELECT d.id, s.id FROM diseases d
JOIN symptoms s ON s.name IN ('fever', 'cough', 'body aches', 'headache', 'fatigue', 'chills')
WHERE d.name = 'Influenza';

INSERT IGNORE INTO disease_symptoms (disease_id, symptom_id)
SELECT d.id, s.id FROM diseases d
JOIN symptoms s ON s.name IN ('headache', 'nausea', 'dizziness', 'sensitivity to light')
WHERE d.name = 'Migraine';

INSERT IGNORE INTO disease_symptoms (disease_id, symptom_id)
SELECT d.id, s.id FROM diseases d
JOIN symptoms s ON s.name IN ('nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'fever')
WHERE d.name = 'Gastroenteritis';