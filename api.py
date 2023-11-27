from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
from pydantic import BaseModel

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medicine_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class MedicineData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    drugName = db.Column(db.String)
    condition = db.Column(db.String)
    rating = db.Column(db.Float)
    usefulCount = db.Column(db.Integer)

vectorizer = joblib.load('tfidfvectorizer_11c.pkl')
model = joblib.load('passmodel_11c.pkl')

class PredictionRequest(BaseModel):
    text: str

@app.route('/top-drugs/<condition>', methods=['GET'])
def top_drugs(condition):
    results = MedicineData.query.filter(MedicineData.rating >= 9,
                                        MedicineData.usefulCount >= 100,
                                        MedicineData.condition == condition)\
                                .order_by(MedicineData.rating.desc(),
                                          MedicineData.usefulCount.desc())\
                                .limit(3)\
                                .all()
    if not results:
        return jsonify({'error': 'No data found'}), 404

    return jsonify([result.drugName for result in results])

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
