# Vercel Python serverless function for sentiment prediction
# Loads both RNN and DistilBERT models for comparison

# Configure PyTorch-only mode (disable TensorFlow backend)
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from http.server import BaseHTTPRequestHandler
import json
import time
import pickle

# Global model cache (persists across function invocations)
_distilbert_model = None
_distilbert_tokenizer = None
_rnn_model = None
_rnn_tokenizer = None

def load_models():
    """Load both DistilBERT and RNN models (cached after first load)."""
    global _distilbert_model, _distilbert_tokenizer, _rnn_model, _rnn_tokenizer
    
    # Load DistilBERT
    if _distilbert_model is None:
        try:
            from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
            import torch
            
            model_path = os.path.join(os.path.dirname(__file__), '../../outputs/transformer')
            _distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            _distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
            _distilbert_model.eval()
            print(f"DistilBERT loaded from {model_path}")
        except Exception as e:
            print(f"Error loading DistilBERT: {e}")
    
    # Load RNN
    if _rnn_model is None:
        try:
            import torch
            import sys
            # Add model paths
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../model_training/model_b'))
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../model_training'))
            from rnn_attention import SentimentRNN
            
            rnn_path = os.path.join(os.path.dirname(__file__), '../../outputs/rnn_sentiment_model.pt')
            tokenizer_path = os.path.join(os.path.dirname(__file__), '../../outputs/rnn_checkpoints/tokenizer.pkl')
            
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                _rnn_tokenizer = pickle.load(f)
            
            # Load model
            checkpoint = torch.load(rnn_path, map_location='cpu')
            vocab_size = checkpoint.get('vocab_size', _rnn_tokenizer.get_vocab_size())
            embedding_dim = checkpoint.get('embedding_dim', 100)
            hidden_dim = checkpoint.get('hidden_dim', 128)
            _rnn_model = SentimentRNN(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=2)
            _rnn_model.load_state_dict(checkpoint['model_state_dict'])
            _rnn_model.eval()
            print(f"RNN loaded from {rnn_path}")
        except Exception as e:
            print(f"Error loading RNN: {e}")
    
    return (_distilbert_model, _distilbert_tokenizer, _rnn_model, _rnn_tokenizer)

def extract_aspects_and_sentiment(text, rnn_model, rnn_tokenizer, distilbert_model, distilbert_tokenizer):
    """
    Extract aspects (topics) and analyze sentiment for each.
    
    Business Value:
    - Identifies specific strengths/weaknesses (food, service, price, etc.)
    - Prioritizes what to improve vs. what to maintain
    - Enables targeted action plans
    """
    import re
    
    # Common aspect keywords for different domains
    ASPECT_KEYWORDS = {
        'food': ['food', 'meal', 'dish', 'taste', 'flavor', 'cuisine', 'menu', 'breakfast', 'lunch', 'dinner'],
        'service': ['service', 'staff', 'waiter', 'waitress', 'employee', 'server', 'host', 'manager'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth', 'affordable'],
        'quality': ['quality', 'fresh', 'clean', 'standard', 'condition'],
        'location': ['location', 'place', 'area', 'convenient', 'parking', 'access'],
        'ambiance': ['atmosphere', 'ambiance', 'ambience', 'decor', 'music', 'vibe', 'environment'],
        'product': ['product', 'item', 'delivery', 'packaging', 'arrived', 'shipping'],
        'experience': ['experience', 'visit', 'time', 'stay']
    }
    
    # Split text into sentences for granular analysis
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    aspect_sentiments = {}
    detailed_insights = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Detect aspects in this sentence
        detected_aspects = []
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(keyword in sentence_lower for keyword in keywords):
                detected_aspects.append(aspect)
        
        if detected_aspects and len(sentence) > 10:  # Ignore very short sentences
            # Analyze sentiment of this sentence
            rnn_result = predict_rnn(sentence, rnn_model, rnn_tokenizer)
            
            for aspect in detected_aspects:
                if aspect not in aspect_sentiments:
                    aspect_sentiments[aspect] = {
                        'mentions': [],
                        'sentiments': [],
                        'sentences': []
                    }
                
                aspect_sentiments[aspect]['mentions'].append(sentence)
                aspect_sentiments[aspect]['sentiments'].append(rnn_result['label'])
                aspect_sentiments[aspect]['sentences'].append({
                    'text': sentence,
                    'sentiment': rnn_result['label'],
                    'confidence': rnn_result['confidence']
                })
    
    # Aggregate aspect sentiments
    aspect_summary = {}
    for aspect, data in aspect_sentiments.items():
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for sent in data['sentiments']:
            sentiment_counts[sent] += 1
        
        total = len(data['sentiments'])
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        aspect_summary[aspect] = {
            'dominant_sentiment': dominant_sentiment,
            'mention_count': total,
            'sentiment_breakdown': sentiment_counts,
            'sample_mentions': data['sentences'][:2],  # Top 2 mentions
            'confidence': sum(s['confidence'] for s in data['sentences']) / total
        }
    
    # Generate business insights
    insights = generate_business_insights(aspect_summary, text)
    
    return {
        'aspects': aspect_summary,
        'insights': insights,
        'total_aspects_detected': len(aspect_summary)
    }

def generate_business_insights(aspect_summary, original_text):
    """Generate actionable business insights from aspect analysis."""
    
    insights = {
        'strengths': [],
        'weaknesses': [],
        'priorities': [],
        'recommendations': [],
        'overall_summary': ''
    }
    
    if not aspect_summary:
        return {
            'strengths': [],
            'weaknesses': [],
            'priorities': [],
            'recommendations': ['Collect more detailed feedback to identify specific aspects'],
            'overall_summary': 'Insufficient aspect-specific feedback for detailed analysis'
        }
    
    # Identify strengths (positive aspects)
    for aspect, data in aspect_summary.items():
        if data['dominant_sentiment'] == 'Positive':
            insights['strengths'].append({
                'aspect': aspect.title(),
                'mentions': data['mention_count'],
                'confidence': f"{data['confidence']:.1%}",
                'action': f"Maintain and promote {aspect} quality"
            })
    
    # Identify weaknesses (negative aspects)
    for aspect, data in aspect_summary.items():
        if data['dominant_sentiment'] == 'Negative':
            insights['weaknesses'].append({
                'aspect': aspect.title(),
                'mentions': data['mention_count'],
                'confidence': f"{data['confidence']:.1%}",
                'action': f"Immediate improvement needed in {aspect}"
            })
    
    # Prioritize improvements (negative aspects with high mention count)
    negative_aspects = [(aspect, data) for aspect, data in aspect_summary.items() 
                       if data['dominant_sentiment'] == 'Negative']
    negative_aspects.sort(key=lambda x: x[1]['mention_count'], reverse=True)
    
    for aspect, data in negative_aspects[:3]:  # Top 3 priorities
        insights['priorities'].append({
            'rank': len(insights['priorities']) + 1,
            'aspect': aspect.title(),
            'urgency': 'High' if data['mention_count'] > 1 else 'Medium',
            'reason': f"Mentioned {data['mention_count']} time(s) negatively"
        })
    
    # Generate recommendations
    if insights['weaknesses']:
        top_weakness = insights['weaknesses'][0]['aspect']
        insights['recommendations'].append(f"🎯 Priority: Address {top_weakness} immediately - highest negative impact")
    
    if insights['strengths']:
        top_strength = insights['strengths'][0]['aspect']
        insights['recommendations'].append(f"✅ Leverage: Promote {top_strength} in marketing - proven strength")
    
    if len(aspect_summary) == 1:
        insights['recommendations'].append("📊 Insight: Feedback is focused on one aspect - encourage broader reviews")
    
    # Overall summary
    pos_count = sum(1 for d in aspect_summary.values() if d['dominant_sentiment'] == 'Positive')
    neg_count = sum(1 for d in aspect_summary.values() if d['dominant_sentiment'] == 'Negative')
    neu_count = sum(1 for d in aspect_summary.values() if d['dominant_sentiment'] == 'Neutral')
    
    if pos_count > neg_count:
        insights['overall_summary'] = f"Overall positive feedback across {pos_count}/{len(aspect_summary)} aspects. Focus on maintaining strengths while addressing {neg_count} weak area(s)."
    elif neg_count > pos_count:
        insights['overall_summary'] = f"Critical attention needed: {neg_count}/{len(aspect_summary)} aspects are negative. Implement immediate improvements."
    else:
        insights['overall_summary'] = f"Mixed feedback detected. Balanced approach needed to enhance {neg_count} weak area(s) and capitalize on {pos_count} strength(s)."
    
    return insights

def predict_distilbert(text, model, tokenizer):
    """Predict sentiment using DistilBERT."""
    import torch
    
    inputs = tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    # Determine label with neutral zone
    NEUTRAL_THRESHOLD = 0.85  # Below this confidence = neutral
    
    if confidence < NEUTRAL_THRESHOLD:
        label = 'Neutral'
        # For neutral, show both probabilities are similar
        confidence = 1.0 - confidence  # Flip to show "confidence in neutrality"
    else:
        label = 'Positive' if prediction == 1 else 'Negative'
    
    return {
        'label': label,
        'confidence': confidence,
        'probabilities': {
            'negative': probabilities[0][0].item(),
            'positive': probabilities[0][1].item()
        }
    }

def predict_rnn(text, model, tokenizer):
    """Predict sentiment using RNN."""
    import torch
    import numpy as np
    
    # Tokenize using encode() method
    sequence = tokenizer.encode(text)
    # Pad
    max_len = 256
    if len(sequence) < max_len:
        sequence = [tokenizer.PAD_IDX] * (max_len - len(sequence)) + sequence
    else:
        sequence = sequence[:max_len]
    
    # Convert to tensor
    input_tensor = torch.LongTensor([sequence])
    
    with torch.no_grad():
        outputs, attention_weights = model(input_tensor)  # Model returns (predictions, attention_weights)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    # Determine label with neutral zone
    NEUTRAL_THRESHOLD = 0.85  # Below this confidence = neutral
    
    if confidence < NEUTRAL_THRESHOLD:
        label = 'Neutral'
        # For neutral, show both probabilities are similar
        confidence = 1.0 - confidence  # Flip to show "confidence in neutrality"
    else:
        label = 'Positive' if prediction == 1 else 'Negative'
    
    return {
        'label': label,
        'confidence': confidence,
        'probabilities': {
            'negative': probabilities[0][0].item(),
            'positive': probabilities[0][1].item()
        }
    }

def predict_sentiment(text, mode='hybrid'):
    """
    Predict sentiment using hybrid ensemble approach.
    
    Args:
        text: Input text to analyze
        mode: 'hybrid', 'compare', 'sequential', 'business-insights', 'rnn-only', 'distilbert-only'
    
    Modes:
        - hybrid: RNN first, DistilBERT verifies uncertain cases (recommended)
        - sequential: RNN detects neutrality, DistilBERT refines pos/neg
        - business-insights: Aspect-based analysis with actionable insights (NEW!)
        - compare: Run both models for comparison
        - rnn-only: Fast RNN predictions only
        - distilbert-only: Accurate DistilBERT only
    """
    distilbert_model, distilbert_tokenizer, rnn_model, rnn_tokenizer = load_models()
    
    CONFIDENCE_THRESHOLD = 0.90  # Route to DistilBERT if RNN < 90% confident
    
    if mode == 'business-insights':
        # BUSINESS INSIGHTS MODE: Aspect-based analysis with actionable recommendations
        # Get overall sentiment first
        overall_rnn = predict_rnn(text, rnn_model, rnn_tokenizer) if rnn_model else None
        overall_distilbert = predict_distilbert(text, distilbert_model, distilbert_tokenizer) if distilbert_model else None
        
        # Smart consensus: Favor DistilBERT (94% accuracy) when models disagree
        if overall_rnn and overall_distilbert:
            if overall_rnn['label'] == overall_distilbert['label']:
                # Models agree - use agreed label with higher confidence
                consensus = overall_rnn['label']
                consensus_confidence = max(overall_rnn['confidence'], overall_distilbert['confidence'])
            else:
                # Models disagree - trust DistilBERT (more accurate model)
                consensus = overall_distilbert['label']
                consensus_confidence = overall_distilbert['confidence']
        elif overall_distilbert:
            consensus = overall_distilbert['label']
            consensus_confidence = overall_distilbert['confidence']
        elif overall_rnn:
            consensus = overall_rnn['label']
            consensus_confidence = overall_rnn['confidence']
        else:
            consensus = 'Unknown'
            consensus_confidence = 0.0
        
        # Extract aspects and generate insights
        aspect_analysis = extract_aspects_and_sentiment(
            text, rnn_model, rnn_tokenizer, distilbert_model, distilbert_tokenizer
        )
        
        return {
            'mode': 'business-insights',
            'overall_sentiment': {
                'rnn': overall_rnn,
                'distilbert': overall_distilbert,
                'consensus': consensus,
                'consensus_confidence': consensus_confidence,
                'models_agreed': overall_rnn['label'] == overall_distilbert['label'] if (overall_rnn and overall_distilbert) else None
            },
            'aspect_analysis': aspect_analysis['aspects'],
            'business_insights': aspect_analysis['insights'],
            'total_aspects': aspect_analysis['total_aspects_detected'],
            'explanation': 'Analyzes specific aspects (food, service, price, etc.) to provide actionable business intelligence'
        }
    
    elif mode == 'sequential':
        # SEQUENTIAL PIPELINE: RNN for ambiguity detection, DistilBERT for refinement
        # Phase 1: RNN analyzes overall sentiment and detects uncertainty
        rnn_result = predict_rnn(text, rnn_model, rnn_tokenizer) if rnn_model else None
        
        if not rnn_result:
            return {'error': 'RNN model not loaded'}
        
        # Phase 2: Decision based on RNN's assessment
        if rnn_result['label'] == 'Neutral':
            # RNN detected ambiguity/uncertainty - TRUST IT
            # Use DistilBERT only to provide additional context if needed
            distilbert_result = predict_distilbert(text, distilbert_model, distilbert_tokenizer) if distilbert_model else None
            
            return {
                'final_prediction': rnn_result,
                'model_used': 'RNN (Neutrality Detector)',
                'reason': 'RNN detected ambiguous/neutral sentiment - RNN excels at uncertainty detection',
                'pipeline_stage': 'Stage 1: RNN Neutrality Detection',
                'rnn_analysis': rnn_result,
                'distilbert_context': distilbert_result,
                'explanation': 'RNN is better at detecting when sentiment is unclear or mixed'
            }
        else:
            # RNN detected clear positive/negative - verify with DistilBERT
            # DistilBERT refines the sentiment classification
            distilbert_result = predict_distilbert(text, distilbert_model, distilbert_tokenizer) if distilbert_model else None
            
            if not distilbert_result:
                return {
                    'final_prediction': rnn_result,
                    'model_used': 'RNN only',
                    'reason': 'DistilBERT not available'
                }
            
            # Use DistilBERT's prediction for clear sentiment (it's more accurate)
            return {
                'final_prediction': distilbert_result,
                'model_used': 'Sequential: RNN → DistilBERT',
                'reason': f'RNN detected clear {rnn_result["label"]} sentiment, DistilBERT refined the analysis',
                'pipeline_stage': 'Stage 2: DistilBERT Refinement',
                'rnn_initial': rnn_result,
                'distilbert_refined': distilbert_result,
                'explanation': 'RNN screened for neutrality, DistilBERT provided nuanced pos/neg classification',
                'agreement': 'Agreed' if rnn_result['label'] == distilbert_result['label'] else 'Disagreed - DistilBERT used'
            }
    
    elif mode == 'hybrid':
        # Original hybrid mode - fast filter
        rnn_result = predict_rnn(text, rnn_model, rnn_tokenizer) if rnn_model else None
        
        if rnn_result and rnn_result['confidence'] >= CONFIDENCE_THRESHOLD:
            return {
                'final_prediction': rnn_result,
                'model_used': 'RNN (Fast Filter)',
                'reason': f"RNN confidence {rnn_result['confidence']:.1%} >= {CONFIDENCE_THRESHOLD:.0%}",
                'rnn_prediction': rnn_result,
                'distilbert_used': False
            }
        else:
            distilbert_result = predict_distilbert(text, distilbert_model, distilbert_tokenizer) if distilbert_model else None
            return {
                'final_prediction': distilbert_result,
                'model_used': 'DistilBERT (Verifier)',
                'reason': f"RNN confidence {rnn_result['confidence']:.1%} < {CONFIDENCE_THRESHOLD:.0%}, routing to DistilBERT",
                'rnn_prediction': rnn_result,
                'distilbert_prediction': distilbert_result,
                'distilbert_used': True
            }
    
    elif mode == 'compare':
        results = {}
        if distilbert_model is not None:
            results['distilbert'] = predict_distilbert(text, distilbert_model, distilbert_tokenizer)
        if rnn_model is not None:
            results['rnn'] = predict_rnn(text, rnn_model, rnn_tokenizer)
        return results
    
    elif mode == 'rnn-only':
        return {'rnn': predict_rnn(text, rnn_model, rnn_tokenizer)} if rnn_model else {}
    
    elif mode == 'distilbert-only':
        return {'distilbert': predict_distilbert(text, distilbert_model, distilbert_tokenizer)} if distilbert_model else {}
    
    return {}

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST request for sentiment prediction."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            # Extract text and mode
            text = data.get('text', '')
            mode = data.get('mode', 'hybrid')  # Default to hybrid mode
            
            if not text:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No text provided'}).encode())
                return
            
            # Predict sentiment (hybrid ensemble by default)
            start_time = time.time()
            predictions = predict_sentiment(text, mode=mode)
            processing_time = int((time.time() - start_time) * 1000)  # ms
            
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
            
        except Exception as e:
            # Error response
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': str(e),
                'message': 'Failed to process request'
            }).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
