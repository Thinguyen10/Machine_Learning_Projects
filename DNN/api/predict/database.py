import sqlite3
import json
from datetime import datetime
import os

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), '../../outputs/sentiment.db')

def clear_all_data():
    """Clear all data from database tables (keeps schema)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Delete all data from tables (cascade will handle foreign keys)
    c.execute('DELETE FROM business_insights')
    c.execute('DELETE FROM aspects')
    c.execute('DELETE FROM sentiment_results')
    c.execute('DELETE FROM reviews')
    c.execute('DELETE FROM upload_batches')
    
    # Reset auto-increment counters
    c.execute('DELETE FROM sqlite_sequence')
    
    conn.commit()
    conn.close()
    print("✓ All database data cleared")

def init_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Reviews table - stores raw reviews and metadata
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            upload_date DATE,
            location VARCHAR(100),
            product VARCHAR(100),
            source VARCHAR(50),
            rating INTEGER,
            metadata TEXT
        )
    ''')
    
    # Sentiment results table - stores analysis results
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER NOT NULL,
            overall_sentiment VARCHAR(20),
            confidence FLOAT,
            model_used VARCHAR(50),
            rnn_sentiment VARCHAR(20),
            rnn_confidence FLOAT,
            distilbert_sentiment VARCHAR(20),
            distilbert_confidence FLOAT,
            analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (review_id) REFERENCES reviews (id)
        )
    ''')
    
    # Aspects table - stores aspect-based sentiment
    c.execute('''
        CREATE TABLE IF NOT EXISTS aspects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER NOT NULL,
            aspect_name VARCHAR(50),
            sentiment VARCHAR(20),
            confidence FLOAT,
            mention_count INTEGER,
            FOREIGN KEY (result_id) REFERENCES sentiment_results (id)
        )
    ''')
    
    # Business insights table - stores generated insights
    c.execute('''
        CREATE TABLE IF NOT EXISTS business_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER NOT NULL,
            strengths TEXT,
            weaknesses TEXT,
            priorities TEXT,
            recommendations TEXT,
            summary TEXT,
            FOREIGN KEY (result_id) REFERENCES sentiment_results (id)
        )
    ''')
    
    # Upload batches table - tracks uploaded files
    c.execute('''
        CREATE TABLE IF NOT EXISTS upload_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename VARCHAR(255),
            upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_rows INTEGER,
            processed_rows INTEGER,
            status VARCHAR(50)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized at {DB_PATH}")

def save_review(text, location=None, product=None, source=None, rating=None, upload_date=None):
    """Save a single review to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO reviews (text, location, product, source, rating, upload_date)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (text, location, product, source, rating, upload_date))
    
    review_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return review_id

def save_sentiment_result(review_id, sentiment_data, mode='sequential'):
    """Save sentiment analysis results"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Extract overall sentiment
    if mode == 'sequential':
        overall_sentiment = sentiment_data['final_prediction']['label']
        confidence = sentiment_data['final_prediction']['confidence']
        rnn_sentiment = sentiment_data['rnn_prediction']['label']
        rnn_confidence = sentiment_data['rnn_prediction']['confidence']
        distilbert_sentiment = sentiment_data.get('distilbert_prediction', {}).get('label')
        distilbert_confidence = sentiment_data.get('distilbert_prediction', {}).get('confidence')
    elif mode == 'business-insights':
        overall_sentiment = sentiment_data['overall_sentiment']['consensus']
        confidence = None
        rnn_sentiment = sentiment_data['overall_sentiment']['rnn']['label']
        rnn_confidence = sentiment_data['overall_sentiment']['rnn']['confidence']
        distilbert_sentiment = sentiment_data['overall_sentiment']['distilbert']['label']
        distilbert_confidence = sentiment_data['overall_sentiment']['distilbert']['confidence']
    else:
        overall_sentiment = sentiment_data.get('label', 'Unknown')
        confidence = sentiment_data.get('confidence')
        rnn_sentiment = None
        rnn_confidence = None
        distilbert_sentiment = None
        distilbert_confidence = None
    
    c.execute('''
        INSERT INTO sentiment_results 
        (review_id, overall_sentiment, confidence, model_used, 
         rnn_sentiment, rnn_confidence, distilbert_sentiment, distilbert_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (review_id, overall_sentiment, confidence, mode,
          rnn_sentiment, rnn_confidence, distilbert_sentiment, distilbert_confidence))
    
    result_id = c.lastrowid
    
    # Save aspects if available
    if mode == 'business-insights' and 'aspect_analysis' in sentiment_data:
        for aspect_name, aspect_data in sentiment_data['aspect_analysis'].items():
            c.execute('''
                INSERT INTO aspects (result_id, aspect_name, sentiment, confidence, mention_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (result_id, aspect_name, 
                  aspect_data['dominant_sentiment'],
                  aspect_data['confidence'],
                  aspect_data['mention_count']))
        
        # Save business insights
        insights = sentiment_data.get('business_insights', {})
        c.execute('''
            INSERT INTO business_insights 
            (result_id, strengths, weaknesses, priorities, recommendations, summary)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (result_id,
              json.dumps(insights.get('strengths', [])),
              json.dumps(insights.get('weaknesses', [])),
              json.dumps(insights.get('improvement_priorities', [])),
              json.dumps(insights.get('recommendations', [])),
              insights.get('overall_summary', '')))
    
    conn.commit()
    conn.close()
    
    return result_id

def create_upload_batch(filename, total_rows):
    """Create a new upload batch record"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO upload_batches (filename, total_rows, processed_rows, status)
        VALUES (?, ?, 0, 'processing')
    ''', (filename, total_rows))
    
    batch_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return batch_id

def update_batch_progress(batch_id, processed_rows, status='processing'):
    """Update upload batch progress"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        UPDATE upload_batches 
        SET processed_rows = ?, status = ?
        WHERE id = ?
    ''', (processed_rows, status, batch_id))
    
    conn.commit()
    conn.close()

def get_recent_batches(limit=10):
    """Get recent upload batches"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT id, filename, upload_timestamp, total_rows, processed_rows, status
        FROM upload_batches
        ORDER BY upload_timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    batches = []
    for row in c.fetchall():
        batches.append({
            'id': row[0],
            'filename': row[1],
            'upload_timestamp': row[2],
            'total_rows': row[3],
            'processed_rows': row[4],
            'status': row[5]
        })
    
    conn.close()
    return batches

def get_sentiment_trends(days=30, location=None):
    """Get sentiment trends over time - aggregated by date for time series visualization"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    query = '''
        SELECT 
            DATE(r.upload_date) as date,
            SUM(CASE WHEN LOWER(sr.overall_sentiment) = 'positive' THEN 1 ELSE 0 END) as positive_count,
            SUM(CASE WHEN LOWER(sr.overall_sentiment) = 'negative' THEN 1 ELSE 0 END) as negative_count,
            SUM(CASE WHEN LOWER(sr.overall_sentiment) IN ('neutral', 'mixed') THEN 1 ELSE 0 END) as neutral_count,
            COUNT(*) as total_count
        FROM reviews r
        JOIN sentiment_results sr ON r.id = sr.review_id
        WHERE r.upload_date >= DATE('now', '-' || ? || ' days')
    '''
    params = [days]
    
    if location:
        query += ' AND r.location = ?'
        params.append(location)
    
    query += '''
        GROUP BY DATE(r.upload_date)
        ORDER BY date ASC
    '''
    
    c.execute(query, params)
    
    trends = []
    for row in c.fetchall():
        total = row[4] if row[4] > 0 else 1  # Avoid division by zero
        trends.append({
            'date': row[0],
            'positive_count': row[1],
            'negative_count': row[2],
            'neutral_count': row[3],
            'total_count': row[4],
            'positive_pct': round((row[1] / total) * 100, 1),
            'negative_pct': round((row[2] / total) * 100, 1),
            'neutral_pct': round((row[3] / total) * 100, 1)
        })
    
    conn.close()
    return trends

def get_aspect_breakdown(location=None, days=30):
    """Get aspect sentiment breakdown - returns positive and negative mentions per aspect"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get sentiment counts per aspect (positive, negative, neutral separately)
    query = '''
        SELECT 
            a.aspect_name,
            SUM(CASE WHEN LOWER(a.sentiment) LIKE '%positive%' THEN a.mention_count ELSE 0 END) as positive_mentions,
            SUM(CASE WHEN LOWER(a.sentiment) LIKE '%negative%' THEN a.mention_count ELSE 0 END) as negative_mentions,
            SUM(CASE WHEN LOWER(a.sentiment) LIKE '%neutral%' THEN a.mention_count ELSE 0 END) as neutral_mentions,
            SUM(a.mention_count) as total_mentions
        FROM aspects a
        JOIN sentiment_results sr ON a.result_id = sr.id
        JOIN reviews r ON sr.review_id = r.id
        WHERE r.upload_date >= DATE('now', '-' || ? || ' days')
    '''
    params = [days]
    
    if location:
        query += ' AND r.location = ?'
        params.append(location)
    
    query += '''
        GROUP BY a.aspect_name
        HAVING total_mentions > 0
        ORDER BY total_mentions DESC
    '''
    
    c.execute(query, params)
    
    aspects = []
    for row in c.fetchall():
        aspect_name = row[0] if row[0] else 'Unknown'
        positive = row[1] or 0
        negative = row[2] or 0
        neutral = row[3] or 0
        total = row[4] or 0
        
        # Determine dominant sentiment
        if positive > negative and positive > neutral:
            dominant = 'positive'
        elif negative > positive and negative > neutral:
            dominant = 'negative'
        else:
            dominant = 'neutral'
        
        aspects.append({
            'aspect_name': aspect_name,
            'dominant_sentiment': dominant,
            'positive_mentions': positive,
            'negative_mentions': negative,
            'neutral_mentions': neutral,
            'total_mentions': total
        })
    
    conn.close()
    return aspects
    
    conn.close()
    return aspects

def get_top_insights(insight_type='strengths', limit=5, location=None):
    """Get top strengths or weaknesses"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    column = insight_type  # 'strengths', 'weaknesses', or 'priorities'
    
    query = f'''
        SELECT 
            bi.{column},
            r.location,
            COUNT(*) as frequency
        FROM business_insights bi
        JOIN sentiment_results sr ON bi.result_id = sr.id
        JOIN reviews r ON sr.review_id = r.id
        WHERE bi.{column} IS NOT NULL AND bi.{column} != '[]'
    '''
    params = []
    
    if location:
        query += ' AND r.location = ?'
        params.append(location)
    
    query += f'''
        GROUP BY bi.{column}
        ORDER BY frequency DESC
        LIMIT ?
    '''
    params.append(limit)
    
    c.execute(query, params)
    
    insights = []
    for row in c.fetchall():
        insights.append({
            'data': json.loads(row[0]) if row[0] else [],
            'location': row[1],
            'frequency': row[2]
        })
    
    conn.close()
    return insights

def get_statistics(location=None, days=30):
    """Get overall statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    query = '''
        SELECT 
            COUNT(*) as total_reviews,
            SUM(CASE WHEN LOWER(sr.overall_sentiment) LIKE '%positive%' THEN 1 ELSE 0 END) as positive_count,
            SUM(CASE WHEN LOWER(sr.overall_sentiment) LIKE '%negative%' THEN 1 ELSE 0 END) as negative_count,
            SUM(CASE WHEN LOWER(sr.overall_sentiment) LIKE '%neutral%' OR LOWER(sr.overall_sentiment) LIKE '%mixed%' THEN 1 ELSE 0 END) as neutral_count,
            AVG(r.rating) as avg_rating,
            AVG(sr.confidence) as avg_confidence
        FROM reviews r
        JOIN sentiment_results sr ON r.id = sr.review_id
        WHERE r.upload_date >= DATE('now', '-' || ? || ' days')
    '''
    params = [days]
    
    if location:
        query += ' AND r.location = ?'
        params.append(location)
    
    c.execute(query, params)
    row = c.fetchone()
    
    total = row[0] or 0
    positive = row[1] or 0
    negative = row[2] or 0
    neutral = row[3] or 0
    
    stats = {
        'total_reviews': total,
        'positive_count': positive,
        'negative_count': negative,
        'neutral_count': neutral,
        'positive_percentage': round((positive / total * 100) if total > 0 else 0, 1),
        'negative_percentage': round((negative / total * 100) if total > 0 else 0, 1),
        'neutral_percentage': round((neutral / total * 100) if total > 0 else 0, 1),
        'avg_rating': round(row[4], 2) if row[4] else None,
        'avg_confidence': round(row[5], 2) if row[5] else None,
        'sentiment_distribution': {
            'positive': round((positive / total * 100) if total > 0 else 0, 1),
            'negative': round((negative / total * 100) if total > 0 else 0, 1),
            'neutral': round((neutral / total * 100) if total > 0 else 0, 1)
        }
    }
    
    conn.close()
    return stats

if __name__ == '__main__':
    # Initialize database when run directly
    init_database()
    print("Database schema created successfully!")
