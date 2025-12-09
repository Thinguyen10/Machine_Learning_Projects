"""
Local development server for sentiment analysis API
Run this for local testing before deploying to Vercel
"""

# Configure PyTorch-only mode (disable TensorFlow backend)
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
import csv
import io
import re
from datetime import datetime
from route import predict_sentiment
from database import (
    init_database, save_review, save_sentiment_result,
    create_upload_batch, update_batch_progress,
    get_recent_batches, get_sentiment_trends,
    get_aspect_breakdown, get_statistics, clear_all_data
)

class LocalHandler(BaseHTTPRequestHandler):
    
    def parse_date(self, date_str):
        """Parse various date formats and return YYYY-MM-DD format"""
        if not date_str or not str(date_str).strip():
            return datetime.now().strftime('%Y-%m-%d')
        
        date_str = str(date_str).strip()
        
        # Common date formats to try
        formats = [
            '%Y-%m-%d',           # 2024-12-08
            '%m/%d/%Y',           # 12/08/2024
            '%d/%m/%Y',           # 08/12/2024
            '%Y/%m/%d',           # 2024/12/08
            '%m-%d-%Y',           # 12-08-2024
            '%d-%m-%Y',           # 08-12-2024
            '%B %d, %Y',          # December 08, 2024
            '%b %d, %Y',          # Dec 08, 2024
            '%Y-%m-%d %H:%M:%S', # 2024-12-08 10:30:00
            '%m/%d/%Y %H:%M:%S', # 12/08/2024 10:30:00
        ]
        
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Try to parse ISO format or other formats
        try:
            # Try ISO format parsing
            if 'T' in date_str:
                parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return parsed_date.strftime('%Y-%m-%d')
        except:
            pass
        
        # If all parsing fails, return today's date
        print(f"⚠️  Could not parse date '{date_str}', using today's date")
        return datetime.now().strftime('%Y-%m-%d')
    
    def detect_columns(self, available_columns):
        """
        Intelligently detect column mappings from various CSV formats.
        Supports common variations like: text/review/comment, sentiment/label, etc.
        """
        columns_lower = {col.lower(): col for col in available_columns}
        
        # Text column variations (most important - the review/comment text)
        text_variations = [
            'review_text', 'text', 'review', 'comment', 'feedback', 'message',
            'content', 'description', 'body', 'reviews', 'sentence', 'tweet',
            'post', 'opinion', 'customerreview', 'user_review'
        ]
        
        # Sentiment/label columns (if pre-labeled dataset)
        sentiment_variations = [
            'sentiment', 'label', 'rating', 'score', 'polarity', 'class',
            'target', 'category', 'emotion', 'feeling'
        ]
        
        # Date columns
        date_variations = [
            'date', 'timestamp', 'time', 'created_at', 'posted_at', 'review_date',
            'created', 'datetime', 'published_date'
        ]
        
        # Location columns
        location_variations = [
            'location', 'city', 'place', 'region', 'branch', 'store',
            'country', 'state', 'area'
        ]
        
        # Product/item columns
        product_variations = [
            'product', 'item', 'product_name', 'service', 'category',
            'item_name', 'product_id', 'sku'
        ]
        
        # Rating columns (numeric 1-5)
        rating_variations = [
            'rating', 'stars', 'score', 'grade', 'rank'
        ]
        
        def find_column(variations):
            for var in variations:
                if var in columns_lower:
                    return columns_lower[var]
            return None
        
        detected = {
            'text': find_column(text_variations),
            'sentiment': find_column(sentiment_variations),
            'date': find_column(date_variations),
            'location': find_column(location_variations),
            'product': find_column(product_variations),
            'rating': find_column(rating_variations)
        }
        
        # If no text column found, use first column with substantial content
        if not detected['text'] and available_columns:
            # Use first column as text by default
            detected['text'] = list(available_columns)[0]
        
        return detected
    
    def extract_text_from_row(self, row, column_map):
        """Extract text content from row using detected column mapping (supports multiple columns)"""
        # Check if multiple text columns are specified
        text_cols = column_map.get('text_columns', [])
        if text_cols:
            # Combine text from all selected columns
            combined_text = []
            for col in text_cols:
                if col in row and row[col]:
                    text = str(row[col]).strip()
                    if text:
                        combined_text.append(text)
            return ' '.join(combined_text) if combined_text else ''
        
        # Fallback to single text column
        text_col = column_map.get('text')
        if text_col and text_col in row:
            return row[text_col].strip()
        
        # Fallback: find first non-empty column with text
        for key, value in row.items():
            if value and len(str(value).strip()) > 10:  # At least 10 chars
                return str(value).strip()
        
        return ''
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle POST request"""
        # Route based on path
        if self.path == '/api/batch-upload':
            self.handle_batch_upload()
        elif self.path == '/api/dashboard':
            self.handle_dashboard_data()
        else:
            self.handle_predict()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/api/batches':
            self.handle_get_batches()
        elif self.path.startswith('/api/stats'):
            self.handle_get_stats()
        else:
            self.send_response(404)
            self.end_headers()
    
    def handle_predict(self):
        """Handle single text prediction"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            # Extract text and mode
            text = data.get('text', '')
            mode = data.get('mode', 'hybrid')
            
            if not text:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No text provided'}).encode())
                return
            
            # Predict sentiment
            start_time = time.time()
            predictions = predict_sentiment(text, mode=mode)
            processing_time = int((time.time() - start_time) * 1000)
            
            # Build response
            result = {
                'predictions': predictions,
                'processing_time': processing_time,
                'text_length': len(text)
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
            print(f"\n✓ Analysis: '{text[:50]}...'")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            self.send_error_response(str(e))
    
    def handle_batch_upload(self):
        """Handle CSV batch upload with flexible column detection"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            csv_data = data.get('csv_data', '')
            filename = data.get('filename', 'upload.csv')
            mode = data.get('mode', 'business-insights')
            user_text_columns = data.get('text_columns', [])  # User-selected text columns (multiple)
            
            if not csv_data:
                self.send_error_response('No CSV data provided', 400)
                return
            
            # Always clear old data before new upload
            print("\n🗑️  Clearing old database data...")
            clear_all_data()
            
            # Parse CSV
            csv_file = io.StringIO(csv_data)
            reader = csv.DictReader(csv_file)
            rows = list(reader)
            
            if not rows:
                self.send_error_response('CSV file is empty', 400)
                return
            
            # Auto-detect column mappings
            column_map = self.detect_columns(rows[0].keys())
            
            # Override text column if user specified one or more
            if user_text_columns and len(user_text_columns) > 0:
                # Validate all columns exist
                valid_columns = [col for col in user_text_columns if col in rows[0].keys()]
                if valid_columns:
                    column_map['text_columns'] = valid_columns
                    print(f"\n✓ Using user-selected text columns: {valid_columns}")
            
            print(f"\n📊 Processing {len(rows)} reviews from '{filename}'...")
            print(f"🔍 Using columns: text={column_map.get('text_columns', column_map.get('text'))}, date='{column_map.get('date')}', location='{column_map.get('location')}'")
            
            # Create batch record
            batch_id = create_upload_batch(filename, len(rows))
            
            results = []
            processed = 0
            
            for idx, row in enumerate(rows):
                try:
                    # Extract data using flexible column mapping
                    text = self.extract_text_from_row(row, column_map)
                    location = row.get(column_map.get('location', ''), '')
                    product = row.get(column_map.get('product', ''), '')
                    rating = row.get(column_map.get('rating', ''))
                    
                    # Parse date from CSV or use today's date
                    date_col = column_map.get('date')
                    raw_date = row.get(date_col, '') if date_col else ''
                    upload_date = self.parse_date(raw_date)
                    
                    if not text.strip():
                        continue
                    
                    # Save review
                    review_id = save_review(
                        text=text,
                        location=location,
                        product=product,
                        source='csv_upload',
                        rating=int(rating) if rating else None,
                        upload_date=upload_date
                    )
                    
                    # Analyze sentiment
                    sentiment = predict_sentiment(text, mode=mode)
                    
                    # Save results
                    result_id = save_sentiment_result(review_id, sentiment, mode)
                    
                    results.append({
                        'review_id': review_id,
                        'result_id': result_id,
                        'sentiment': sentiment.get('final_prediction', {}).get('label') or 
                                   sentiment.get('overall_sentiment', {}).get('consensus', 'Unknown')
                    })
                    
                    processed += 1
                    
                    # Update progress every 10 reviews
                    if processed % 10 == 0:
                        update_batch_progress(batch_id, processed)
                        print(f"  Progress: {processed}/{len(rows)}")
                
                except Exception as e:
                    print(f"  ✗ Error processing row {idx}: {e}")
                    continue
            
            # Mark batch as complete
            update_batch_progress(batch_id, processed, 'completed')
            
            # Send response
            response = {
                'batch_id': batch_id,
                'total_rows': len(rows),
                'processed_rows': processed,
                'results': results[:10],  # Return first 10 for preview
                'message': f'Successfully processed {processed}/{len(rows)} reviews'
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
            print(f"✓ Batch complete: {processed} reviews processed\n")
            
        except Exception as e:
            print(f"✗ Batch upload error: {e}")
            self.send_error_response(str(e))
    
    def handle_dashboard_data(self):
        """Handle dashboard data request"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            data = json.loads(body)
            
            location = data.get('location')
            days = data.get('days', 30)
            
            # Get all dashboard data
            dashboard = {
                'statistics': get_statistics(location, days),
                'trends': get_sentiment_trends(days, location),
                'aspects': get_aspect_breakdown(location, days),
                'recent_batches': get_recent_batches(5)
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(dashboard).encode())
            
        except Exception as e:
            print(f"✗ Dashboard error: {e}")
            self.send_error_response(str(e))
    
    def handle_get_batches(self):
        """Get recent upload batches"""
        try:
            batches = get_recent_batches(10)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'batches': batches}).encode())
            
        except Exception as e:
            self.send_error_response(str(e))
    
    def handle_get_stats(self):
        """Get statistics"""
        try:
            stats = get_statistics()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())
            
        except Exception as e:
            self.send_error_response(str(e))
    
    def send_error_response(self, error_message, code=500):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            'error': error_message,
            'message': 'Failed to process request'
        }).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

if __name__ == '__main__':
    PORT = 8000
    server = HTTPServer(('localhost', PORT), LocalHandler)
    print(f"🚀 Sentiment Analysis API running at http://localhost:{PORT}")
    print(f"📍 Endpoint: POST http://localhost:{PORT}/api/predict")
    print(f"\\n💡 Test with:")
    print(f'   curl -X POST http://localhost:{PORT}/api/predict \\\\')
    print(f'     -H "Content-Type: application/json" \\\\')
    print(f'     -d \'{{\"text\": \"This is amazing!\"}}\'')
    print(f"\\nPress Ctrl+C to stop\\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\n\\n👋 Server stopped")
        server.shutdown()
