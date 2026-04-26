from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
import joblib
from nltk.corpus import stopwords
import sqlite3
from datetime import datetime

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

print("Loading models...")

ml_model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
print("Models loaded successfully!")

stop_words = set(stopwords.words('english'))

# Database setup
def init_db():
    conn = sqlite3.connect('spam_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_prediction(message, prediction, confidence):
    conn = sqlite3.connect('spam_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (message, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (message, prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

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
    
    prediction = 'SPAM' if pred == 1 else 'HAM'
    confidence = round(float(prob), 4)
    
    # Save to database
    save_prediction(message, prediction, confidence)
    
    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'model': 'Logistic Regression'
    })

@app.route('/stats', methods=['GET'])
def stats():
    conn = sqlite3.connect('spam_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'SPAM'")
    spam_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'HAM'")
    ham_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(confidence) FROM predictions WHERE prediction = 'SPAM'")
    avg_spam_conf = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT AVG(confidence) FROM predictions WHERE prediction = 'HAM'")
    avg_ham_conf = cursor.fetchone()[0] or 0
    
    # Last 5 predictions
    cursor.execute('''
        SELECT message, prediction, confidence, timestamp 
        FROM predictions 
        ORDER BY id DESC 
        LIMIT 5
    ''')
    recent = cursor.fetchall()
    
    conn.close()
    
    recent_list = []
    for r in recent:
        recent_list.append({
            'message': r[0][:50] + '...' if len(r[0]) > 50 else r[0],
            'prediction': r[1],
            'confidence': round(r[2] * 100, 1),
            'time': r[3]
        })
    
    return jsonify({
        'total_predictions': total,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'spam_percentage': round(spam_count/total*100, 1) if total > 0 else 0,
        'avg_spam_confidence': round(avg_spam_conf * 100, 1),
        'avg_ham_confidence': round(avg_ham_conf * 100, 1),
        'recent_predictions': recent_list
    })

@app.route('/health')
def health():
    return 'OK'

if __name__ == '__main__':
    print("Server starting at http://127.0.0.1:5000")
    print("Stats endpoint: http://127.0.0.1:5000/stats")
    app.run(debug=True, host='0.0.0.0', port=5000)