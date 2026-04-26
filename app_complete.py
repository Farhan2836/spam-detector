from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
import pickle
import joblib
from nltk.corpus import stopwords
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

print("="*40)
print("Loading models...")
print("="*40)

# Load ML Model
ml_model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
print("[OK] ML Model (Logistic Regression) loaded")

# Load LSTM Model
try:
    lstm_model = load_model('spam_detector_lstm.h5', compile=False)
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    MAX_LEN = 70
    print("[OK] LSTM Model loaded")
    lstm_available = True
except Exception as e:
    print(f"[WARN] LSTM not available: {e}")
    lstm_available = False

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return jsonify({
        'status': 'ok',
        'models': ['Logistic Regression', 'LSTM'] if lstm_available else ['Logistic Regression']
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    cleaned = clean_text(message)
    
    # ML Prediction
    vector = tfidf.transform([cleaned])
    ml_pred = ml_model.predict(vector)[0]
    ml_conf = float(ml_model.predict_proba(vector)[0][1])
    
    result = {
        'message': message,
        'logistic_regression': {
            'prediction': 'SPAM' if ml_pred == 1 else 'HAM',
            'confidence': round(ml_conf, 4)
        }
    }
    
    # LSTM Prediction
    if lstm_available:
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        lstm_conf = float(lstm_model.predict(padded, verbose=0)[0][0])
        result['lstm'] = {
            'prediction': 'SPAM' if lstm_conf > 0.5 else 'HAM',
            'confidence': round(lstm_conf, 4)
        }
    
    return jsonify(result)

@app.route('/predict/ml', methods=['POST'])
def predict_ml_only():
    data = request.get_json()
    message = data.get('message', '')
    cleaned = clean_text(message)
    vector = tfidf.transform([cleaned])
    pred = ml_model.predict(vector)[0]
    prob = ml_model.predict_proba(vector)[0][1]
    return jsonify({
        'prediction': 'SPAM' if pred == 1 else 'HAM',
        'confidence': round(float(prob), 4),
        'model': 'Logistic Regression'
    })

@app.route('/predict/lstm', methods=['POST'])
def predict_lstm_only():
    if not lstm_available:
        return jsonify({'error': 'LSTM not available'}), 503
    data = request.get_json()
    message = data.get('message', '')
    cleaned = clean_text(message)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = float(lstm_model.predict(padded, verbose=0)[0][0])
    return jsonify({
        'prediction': 'SPAM' if prob > 0.5 else 'HAM',
        'confidence': round(prob, 4),
        'model': 'LSTM'
    })

if __name__ == '__main__':
    print("="*40)
    print("Starting Spam Detection API")
    print("ML + LSTM both loaded!" if lstm_available else "ML only")
    print("="*40)
    app.run(debug=True, host='0.0.0.0', port=5000)