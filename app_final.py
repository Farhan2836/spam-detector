from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
import joblib
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

print("="*40)
print("Loading models...")
print("="*40)

# Load ML Model
ml_model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
print("OK - Logistic Regression model loaded")
print("OK - TF-IDF Vectorizer loaded")

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
        'message': 'Spam Detection API is running!',
        'model': 'Logistic Regression',
        'accuracy': '98.3%'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        cleaned = clean_text(message)
        vector = tfidf.transform([cleaned])
        pred = ml_model.predict(vector)[0]
        prob = ml_model.predict_proba(vector)[0][1]
        
        return jsonify({
            'success': True,
            'message': message,
            'prediction': 'SPAM' if pred == 1 else 'HAM',
            'confidence': round(float(prob), 4),
            'model': 'Logistic Regression'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return 'OK'

if __name__ == '__main__':
    print("="*40)
    print("Server starting at http://127.0.0.1:5000")
    print("="*40)
    app.run(debug=True, host='0.0.0.0', port=5000)