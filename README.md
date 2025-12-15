# PulseSense – AI-Powered Health Diagnostics

PulseSense is an AI-assisted health diagnostics platform that combines a **FastAPI backend**, **MySQL persistence**, and a **Streamlit frontend**. It enables users to securely manage medical records, perform symptom-based health checks using a lightweight RAG (Retrieval-Augmented Generation) pipeline, and interact with a health-focused chatbot.

This project is designed for both **end users** (to run and explore the application) and **developers** (to understand, extend, and contribute to the system).

---

## Key Features

* Secure authentication (register/login) using **JWT bearer tokens** and **bcrypt password hashing**
* Personal medical record management (create, view, delete)
* AI-powered symptom checker with:

  * Disease prediction
  * Confidence score
  * Recommended specialist
  * Suggested medical tests
  * Precautionary advice
* Health chatbot with conversation history stored per user
* User dashboard and profile view built with Streamlit
* Automatic database bootstrapping with seed medical data
* CORS-enabled backend for local development

---

## Tech Stack

**Backend**

* Python 3.10+
* FastAPI
* SQLAlchemy
* MySQL 8.x
* JWT (Authentication)
* Sentence Transformers / Transformers (optional)
* FAISS (optional, with graceful fallback)

**Frontend**

* Streamlit

**AI / NLP**

* Lightweight RAG pipeline
* Rule-based fallback for reliability

---

## Project Structure

```
PulseSense-AI-Based-Health-Diagnostics/
│
├── backend/          # FastAPI backend (auth, RAG agent, DB access)
│   └── app/
│       └── main.py
│
├── frontend/         # Streamlit frontend
│   └── app.py
│
├── database/         # SQL bootstrap scripts and DB helpers
│   └── init.sql
│
└── README.md
```

---

## Prerequisites

* Python 3.10 or higher
* MySQL 8.x (local or remote)
* Internet connection (for first-time model downloads)
* GPU is optional (CPU fallback supported)

> **Note:** Node.js is *not required*.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Naiv-g/PulseSense-AI-Based-Health-Diagnostics.git
cd PulseSense-AI-Based-Health-Diagnostics
```

---

### 2. MySQL Setup

Create a database:

```sql
CREATE DATABASE pulsesense;
```

You can initialize the database in one of two ways:

**Option A – Manual initialization**

```bash
python database/run_init_sql.py
```

**Option B – Automatic bootstrap (recommended)**
The backend automatically creates tables and inserts sample medical data on startup.

---

### 3. Environment Configuration

Create a `.env` file inside the `backend/` directory:

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=pulsesense
SECRET_KEY=replace-with-a-strong-secret
```

---

### 4. Run the Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS

pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

* API Docs: `http://localhost:8000/docs`

---

### 5. Run the Frontend

```bash
cd frontend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The frontend connects to the backend via:

```
http://localhost:8000
```

(Editable in `frontend/app.py` as `BACKEND_URL`.)

---

## API Overview (Selected Endpoints)

| Method | Endpoint                | Description                      |
| ------ | ----------------------- | -------------------------------- |
| POST   | `/register`             | Create a new user                |
| POST   | `/login`                | Authenticate and receive JWT     |
| GET    | `/medical-records`      | List user medical records        |
| POST   | `/medical-records`      | Create a new medical record      |
| DELETE | `/medical-records/{id}` | Delete a record                  |
| GET    | `/symptoms`             | Fetch seed symptoms              |
| POST   | `/predict-disease`      | Symptom-based disease prediction |
| POST   | `/chatbot/query`        | Health chatbot query             |
| GET    | `/rag-status`           | RAG system diagnostics           |

All protected endpoints require:

```
Authorization: Bearer <access_token>
```

---

## RAG / AI Architecture Notes

* Uses sentence-transformer embeddings with FAISS when available
* Falls back to an in-memory vector store if FAISS is unavailable
* Transformer models (e.g., `google/flan-t5-*`) download on first run
* If NLP dependencies fail, the system gracefully degrades to a rule-based engine

This ensures the application remains functional even in constrained environments.

---

## Development & Contribution

Contributions are welcome.

1. Fork the repository
2. Create a feature branch (`feature/your-feature-name`)
3. Commit changes with clear messages
4. Push and open a Pull Request

Please ensure:

* Code is clean and readable
* APIs are documented where applicable
* No credentials are committed

---

## Security Notes

* Replace `SECRET_KEY` in production
* Restrict CORS origins before deployment
* Do not use root DB credentials in production

---

## Troubleshooting

* **MySQL access denied**: Check `.env` credentials and ensure MySQL is running
* **Slow model download**: Pre-download models or ensure stable internet
* **CORS errors**: Update CORS settings in `backend/app/main.py`

---

## Disclaimer

PulseSense is **not a medical device** and should not be used for clinical diagnosis or treatment decisions. It is intended strictly for educational and research purposes.

---

## Author

Developed by **Naiv-g**
GitHub: [https://github.com/Naiv-g](https://github.com/Naiv-g)

---

## License

This project is licensed under the **MIT License** (or update if different).
