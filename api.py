# To run the server:
# uvicorn your_script_name:app --reload

from fastapi import FastAPI, HTTPException
import sqlite3
import joblib
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    text: str


app = FastAPI()


def top_drugs_extractor(condition):
    db_file = 'medicine_data.db'
    query = """
    SELECT drugName
    FROM medicine_data
    WHERE rating >= 9 AND usefulCount >= 100 AND condition = ?
    ORDER BY rating DESC, usefulCount DESC
    LIMIT 3;
    """

    conn = sqlite3.connect(db_file)
    try:
        cursor = conn.cursor()
        cursor.execute(query, (condition,))
        results = cursor.fetchall()
        return [result[0] for result in results]
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


vectorizer = joblib.load('tfidfvectorizer_11c.pkl')
model = joblib.load('passmodel_11c.pkl')


@app.get("/top-drugs/{condition}")
async def top_drugs(condition: str):
    return top_drugs_extractor(condition)


@app.post("/predict")
async def predict(request: PredictionRequest):
    test_vector = vectorizer.transform([request.text])
    prediction = model.predict(test_vector)
    return {"prediction": prediction[0]}

