"""
Sentiment Analysis Dashboard - Streamlit App
Uses your trained DistilBERT model from HuggingFace
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="AI Sentiment Analysis Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Model configuration
MODEL_ID = "Thi144/sentiment-distilbert"

@st.cache_resource
def load_model():
    """Load model from HuggingFace (cached for performance)"""
    with st.spinner("🔄 Loading AI model from HuggingFace..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        model.eval()
    return tokenizer, model

def analyze_text(text, tokenizer, model):
    """Analyze sentiment of a single text with neutral detection"""
    if not text or len(text.strip()) < 3:
        return {
            "label": "neutral",
            "confidence": 0.5,
            "positive_score": 0.33,
            "negative_score": 0.33,
            "neutral_score": 0.34
        }
    
    # Rule-based neutral keywords check first
    neutral_keywords = [
        'average', 'okay', 'decent', 'fine', 'acceptable', 'moderate', 'fair',
        'mixed', 'so-so', 'mediocre', 'neither', 'neutral', 'balanced'
    ]
    text_lower = text.lower()
    has_neutral_keywords = any(keyword in text_lower for keyword in neutral_keywords)
    
    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    negative_score = predictions[0][0].item()
    positive_score = predictions[0][1].item()
    
    # Calculate score difference
    score_diff = abs(positive_score - negative_score)
    max_score = max(positive_score, negative_score)
    
    # Enhanced neutral detection with multiple conditions
    is_neutral = False
    
    # Condition 1: Has explicit neutral language
    if has_neutral_keywords and score_diff < 0.35:
        is_neutral = True
    
    # Condition 2: Scores are very close (within 20%)
    elif score_diff < 0.20:
        is_neutral = True
    
    # Condition 3: Neither score is confident (both under 60%)
    elif max_score < 0.60:
        is_neutral = True
    
    # Condition 4: Scores in the middle range (45-55% for both)
    elif 0.45 <= positive_score <= 0.55 and 0.45 <= negative_score <= 0.55:
        is_neutral = True
    
    if is_neutral:
        neutral_score = max(0.6, 1.0 - score_diff)
        label = "neutral"
        confidence = neutral_score
    else:
        # Clear positive or negative
        label = "positive" if positive_score > negative_score else "negative"
        confidence = max_score
        neutral_score = 1.0 - confidence
    
    return {
        "label": label,
        "confidence": confidence,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "neutral_score": neutral_score
    }

def extract_aspects(text, sentiment, confidence):
    """Extract aspects (topics) from text and analyze sentiment"""
    text_lower = text.lower()
    
    aspect_keywords = {
        'food': ['food', 'meal', 'dish', 'taste', 'flavor', 'cuisine', 'menu', 'breakfast', 'lunch', 'dinner'],
        'service': ['service', 'staff', 'waiter', 'employee', 'server', 'host', 'manager'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth', 'affordable'],
        'quality': ['quality', 'fresh', 'clean', 'standard', 'condition'],
        'location': ['location', 'place', 'area', 'convenient', 'parking', 'access'],
        'ambiance': ['atmosphere', 'ambiance', 'decor', 'music', 'vibe', 'environment'],
        'product': ['product', 'item', 'delivery', 'packaging', 'shipping'],
        'experience': ['experience', 'visit', 'time', 'stay']
    }
    
    detected_aspects = []
    for aspect, keywords in aspect_keywords.items():
        mentions = sum(1 for keyword in keywords if keyword in text_lower)
        if mentions > 0:
            detected_aspects.append({
                'aspect': aspect,
                'sentiment': sentiment,
                'confidence': confidence,
                'mentions': mentions
            })
    
    return detected_aspects

def main():
    # Header
    st.title("🤖 AI Sentiment Analysis Dashboard")
    st.markdown("**Powered by DistilBERT Transformer Model**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Options")
        analysis_mode = st.radio(
            "Choose Mode:",
            ["Single Text Analysis", "Batch CSV Upload"]
        )
        
        st.markdown("---")
        st.markdown("### 🔬 Model Info")
        st.info(f"**Model:** {MODEL_ID}\n\n**Accuracy:** ~94%")
    
    # Load model
    tokenizer, model = load_model()
    st.success("✅ Model loaded and ready!")
    
    if analysis_mode == "Single Text Analysis":
        # Single text analysis
        st.header("📝 Analyze Single Text")
        
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your review, comment, or feedback here...",
            height=150
        )
        
        if st.button("🚀 Analyze Sentiment", type="primary"):
            if text_input:
                with st.spinner("Analyzing..."):
                    result = analyze_text(text_input, tokenizer, model)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    emoji = "😊" if result["label"] == "positive" else ("😐" if result["label"] == "neutral" else "😞")
                    st.metric("Sentiment", f"{emoji} {result['label'].title()}")
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                with col3:
                    st.metric("Text Length", f"{len(text_input)} chars")
                
                # Probability bars
                st.subheader("📊 Detailed Scores")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**😊 Positive**")
                    st.progress(result['positive_score'])
                    st.write(f"{result['positive_score']:.1%}")
                
                with col2:
                    st.markdown("**😐 Neutral**")
                    st.progress(result['neutral_score'])
                    st.write(f"{result['neutral_score']:.1%}")
                
                with col3:
                    st.markdown("**😞 Negative**")
                    st.progress(result['negative_score'])
                    st.write(f"{result['negative_score']:.1%}")
            else:
                st.warning("⚠️ Please enter some text to analyze")
    
    else:
        # Batch CSV upload
        st.header("📊 Batch Analysis from CSV")
        
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file containing reviews or text to analyze"
        )
        
        if uploaded_file:
            # Load CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            # Show preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head(10))
            
            # Column selection
            st.subheader("⚙️ Configuration")
            text_columns = st.multiselect(
                "Select column(s) containing text to analyze:",
                df.columns.tolist(),
                default=[df.columns[0]] if len(df.columns) > 0 else []
            )
            
            if not text_columns:
                st.warning("⚠️ Please select at least one column to analyze")
                return
            
            # Process button
            if st.button("🚀 Analyze All Reviews", type="primary"):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                total = len(df)
                
                # Process each row
                for idx, row in df.iterrows():
                    # Combine text from all selected columns
                    combined_text = " ".join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                    sentiment = analyze_text(combined_text, tokenizer, model)
                    aspects = extract_aspects(combined_text, sentiment['label'], sentiment['confidence'])
                    
                    results.append({
                        **row.to_dict(),
                        'combined_text': combined_text,
                        'sentiment': sentiment['label'],
                        'confidence': sentiment['confidence'],
                        'positive_score': sentiment['positive_score'],
                        'negative_score': sentiment['negative_score'],
                        'neutral_score': sentiment['neutral_score'],
                        'aspects': aspects
                    })
                    
                    # Update progress
                    progress = (idx + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {idx + 1}/{total} reviews...")
                
                status_text.text("✅ Analysis complete!")
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                # Summary statistics
                st.header("📈 Analysis Summary")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                positive_count = (results_df['sentiment'] == 'positive').sum()
                neutral_count = (results_df['sentiment'] == 'neutral').sum()
                negative_count = (results_df['sentiment'] == 'negative').sum()
                avg_confidence = results_df['confidence'].mean()
                
                with col1:
                    st.metric("Total Reviews", total)
                with col2:
                    st.metric("😊 Positive", f"{positive_count} ({positive_count/total*100:.1f}%)")
                with col3:
                    st.metric("😐 Neutral", f"{neutral_count} ({neutral_count/total*100:.1f}%)")
                with col4:
                    st.metric("😞 Negative", f"{negative_count} ({negative_count/total*100:.1f}%)")
                with col5:
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                # Visualizations
                st.subheader("📊 Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    sentiment_counts = results_df['sentiment'].value_counts()
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'positive': '#4CAF50', 
                            'neutral': '#9E9E9E',
                            'negative': '#F44336'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Confidence distribution
                    fig_hist = px.histogram(
                        results_df,
                        x='confidence',
                        nbins=20,
                        title="Confidence Score Distribution",
                        labels={'confidence': 'Confidence Score'},
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Aspect Analysis
                st.subheader("🏷️ Top Aspects Mentioned")
                
                # Aggregate all aspects
                all_aspects = []
                for aspects_list in results_df['aspects']:
                    if isinstance(aspects_list, list):
                        all_aspects.extend(aspects_list)
                
                if all_aspects:
                    aspect_df = pd.DataFrame(all_aspects)
                    
                    # Group by aspect and sentiment
                    aspect_summary = aspect_df.groupby(['aspect', 'sentiment']).agg({
                        'mentions': 'sum',
                        'confidence': 'mean'
                    }).reset_index()
                    
                    # Create aspect visualization
                    fig_aspects = px.bar(
                        aspect_summary,
                        x='aspect',
                        y='mentions',
                        color='sentiment',
                        title="Aspect Mentions by Sentiment",
                        labels={'mentions': 'Number of Mentions', 'aspect': 'Aspect'},
                        color_discrete_map={
                            'positive': '#4CAF50',
                            'neutral': '#9E9E9E',
                            'negative': '#F44336'
                        },
                        barmode='group'
                    )
                    st.plotly_chart(fig_aspects, use_container_width=True)
                    
                    # Top aspects table
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🌟 Strengths (Positive Aspects)**")
                        positive_aspects = aspect_summary[aspect_summary['sentiment'] == 'positive'].nlargest(5, 'mentions')
                        if not positive_aspects.empty:
                            for _, row in positive_aspects.iterrows():
                                st.success(f"**{row['aspect'].title()}**: {int(row['mentions'])} mentions ({row['confidence']:.1%} confidence)")
                        else:
                            st.info("No positive aspects detected")
                    
                    with col2:
                        st.markdown("**⚠️ Areas for Improvement (Negative Aspects)**")
                        negative_aspects = aspect_summary[aspect_summary['sentiment'] == 'negative'].nlargest(5, 'mentions')
                        if not negative_aspects.empty:
                            for _, row in negative_aspects.iterrows():
                                st.error(f"**{row['aspect'].title()}**: {int(row['mentions'])} mentions ({row['confidence']:.1%} confidence)")
                        else:
                            st.info("No negative aspects detected")
                else:
                    st.info("No specific aspects detected in the reviews. Reviews may be too general or short.")
                
                # Show results table
                st.subheader("📋 Detailed Results")
                
                # Add filters
                filter_sentiment = st.multiselect(
                    "Filter by sentiment:",
                    ['positive', 'neutral', 'negative'],
                    default=['positive', 'neutral', 'negative']
                )
                
                filtered_df = results_df[results_df['sentiment'].isin(filter_sentiment)]
                
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=400
                )
                
                # Download results
                st.subheader("💾 Download Results")
                
                # Convert to CSV
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=f"sentiment_analysis_{timestamp}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
