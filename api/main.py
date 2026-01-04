import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import psycopg2

app = FastAPI(title="AI Spam Detection API")

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", 5432)
    )

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(data: TextInput):
    text_vector = vectorizer.transform([data.text])
    probability = model.predict_proba(text_vector)[0][1]
    result = "Spam" if probability > 0.5 else "Not Spam"

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO public.predictions (input_text, prediction, confidence) VALUES (%s, %s, %s)",
        (data.text, result, float(probability))
    )

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "input_text": data.text,
        "prediction": result,
        "spam_confidence": round(float(probability) * 100, 2)
    }
