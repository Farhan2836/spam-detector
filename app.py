from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
import joblib
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

print("Loading models...")

ml_model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
print("Models loaded successfully!")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return 'Spam Detection API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    cleaned = clean_text(message)
    vector = tfidf.transform([cleaned])
    pred = ml_model.predict(vector)[0]
    prob = ml_model.predict_proba(vector)[0][1]
    return jsonify({
        'prediction': 'SPAM' if pred == 1 else 'HAM',
        'confidence': round(float(prob), 4)
    })

@app.route('/health')
def health():
    return 'OK'

if __name__ == '__main__':
    print("Server starting at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)