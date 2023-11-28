from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

app = Flask(__name__)


vectorizer = joblib.load('tfidfvectorizer_11c.pkl')
model = joblib.load('passmodel_11c.pkl')
data = pd.read_table('drugsComTrain_raw.tsv')

class PredictionRequest(BaseModel):
    text: str

def top_drugs_extractor(condition):
    df_top = data[(data['rating']>=9)&(data['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst

@app.route('/top-drugs/<condition>', methods=['GET'])
def top_drugs(condition):
    df_top = data[(data['rating']>=9)&(data['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    results = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    
    if not results:
        return jsonify({'error': 'No data found'}), 404

    return jsonify(results)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    try:
        request_obj = PredictionRequest(text=data['text'])
        test_vector = vectorizer.transform([request_obj.text])
        prediction = model.predict(test_vector)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
