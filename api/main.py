from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import psycopg2

# 1️⃣ Create FastAPI app FIRST
app = FastAPI(title="AI Spam Detection API")

# 2️⃣ Add CORS middleware AFTER app is created
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (ok for learning)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3️⃣ Load ML model
model = joblib.load("../model/spam_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

# 4️⃣ Database connection function
def get_db_connection():
    return psycopg2.connect(
        host="db.ndftnttxjwflywterduv.supabase.co",
        database="postgres",
        user="postgres",
        password="Supabase@2026",
        port=5432
    )

# 5️⃣ Input schema
class TextInput(BaseModel):
    text: str

# 6️⃣ API endpoint
@app.post("/predict")
def predict_spam(data: TextInput):
    text_vector = vectorizer.transform([data.text])

    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0][1]

    result = "Spam" if probability > 0.5 else "Not Spam"

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (input_text, prediction, confidence)
            VALUES (%s, %s, %s)
            """,
            (data.text, result, float(probability))
        )

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print("DB ERROR:", e)
        return {"error": "Database insert failed"}

    return {
        "input_text": data.text,
        "prediction": result,
        "spam_confidence": round(float(probability) * 100, 2)
    }
