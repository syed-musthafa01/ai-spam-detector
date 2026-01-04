import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import psycopg2
from fastapi.middleware.cors import CORSMiddleware

# -------------------- APP --------------------
app = FastAPI(title="AI Spam Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "model", "vectorizer.pkl")

# -------------------- LOAD MODEL --------------------
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# -------------------- DATABASE --------------------
def get_db_connection():
    try:
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "db.ndftnttxjwflywterduv.supabase.co"),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "Supabase@2026"),
            port=os.getenv("DB_PORT", "5432"),
        )
    except Exception as e:
        print("Database connection error:", e)
        return None

# -------------------- SCHEMA --------------------
class TextInput(BaseModel):
    text: str

# -------------------- API --------------------
@app.post("/predict")
def predict_spam(data: TextInput):
    try:
        # ML prediction
        text_vector = vectorizer.transform([data.text])
        probability = model.predict_proba(text_vector)[0][1]
        result = "Spam" if probability > 0.5 else "Not Spam"

        # DB insert
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO public.predictions
                (input_text, prediction, confidence)
                VALUES (%s, %s, %s)
                """,
                (data.text, result, float(probability))
            )
            conn.commit()
            cursor.close()
            conn.close()
        else:
            print("Skipping DB insert (DB not connected)")

        return {
            "input_text": data.text,
            "prediction": result,
            "spam_confidence": round(float(probability) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
