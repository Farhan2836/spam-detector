import sqlite3
from datetime import datetime

DB_NAME = 'spam_predictions.db'

def create_table():
    """Create predictions table if not exists"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            cleaned_message TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT DEFAULT 'Logistic Regression',
            user_feedback TEXT,
            ip_address TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database table created successfully")

def save_prediction(message, cleaned_message, prediction, confidence, model_used="Logistic Regression", ip_address=None):
    """Save prediction to database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (message, cleaned_message, prediction, confidence, model_used, ip_address, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (message, cleaned_message, prediction, confidence, model_used, ip_address, datetime.now()))
    
    conn.commit()
    conn.close()
    return cursor.lastrowid

def update_feedback(prediction_id, feedback):
    """Update user feedback for a prediction"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE predictions 
        SET user_feedback = ? 
        WHERE id = ?
    ''', (feedback, prediction_id))
    
    conn.commit()
    conn.close()
    return True

def get_all_predictions(limit=50):
    """Get recent predictions"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, message, prediction, confidence, timestamp, user_feedback 
        FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    conn.close()
    
    return [{
        'id': r[0],
        'message': r[1][:100] + ('...' if len(r[1]) > 100 else ''),
        'prediction': r[2],
        'confidence': round(r[3] * 100, 1),
        'timestamp': r[4],
        'feedback': r[5]
    } for r in results]

def get_stats():
    """Get statistics from database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    
    # Today's predictions
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE DATE(timestamp) = DATE('now')")
    today = cursor.fetchone()[0]
    
    # Spam vs Ham count
    cursor.execute("SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction")
    counts = dict(cursor.fetchall())
    
    # Average confidence
    cursor.execute("SELECT AVG(confidence) FROM predictions WHERE prediction = 'SPAM'")
    avg_spam_conf = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT AVG(confidence) FROM predictions WHERE prediction = 'HAM'")
    avg_ham_conf = cursor.fetchone()[0] or 0
    
    # Feedback stats
    cursor.execute("SELECT user_feedback, COUNT(*) FROM predictions WHERE user_feedback IS NOT NULL GROUP BY user_feedback")
    feedback_counts = dict(cursor.fetchall())
    
    conn.close()
    
    return {
        'total_predictions': total,
        'today_predictions': today,
        'spam_count': counts.get('SPAM', 0),
        'ham_count': counts.get('HAM', 0),
        'spam_percentage': round((counts.get('SPAM', 0) / total * 100), 1) if total > 0 else 0,
        'avg_spam_confidence': round(avg_spam_conf * 100, 1),
        'avg_ham_confidence': round(avg_ham_conf * 100, 1),
        'feedback_correct': feedback_counts.get('correct', 0),
        'feedback_incorrect': feedback_counts.get('incorrect', 0)
    }

def delete_prediction(prediction_id):
    """Delete a prediction by ID"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    
    conn.commit()
    conn.close()
    return True

if __name__ == '__main__':
    create_table()
    print("Database setup complete!")