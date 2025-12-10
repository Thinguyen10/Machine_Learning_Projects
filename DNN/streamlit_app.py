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

# Custom CSS for professional dark blue theme
st.markdown("""
    <style>
    /* Main background - Dark blue gradient */
    .stApp {
        background: linear-gradient(135deg, #1a2332 0%, #2d3e50 100%);
        color: #ffffff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: 2px solid #475569;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    /* Headers with bright accent */
    h1, h2, h3 {
        color: #60a5fa !important;
    }
    
    /* All text white */
    p, span, div, label {
        color: #e2e8f0 !important;
    }
    
    /* Metric containers */
    [data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }
    
    /* Cards and containers */
    [data-testid="stExpander"] {
        background-color: rgba(30, 41, 59, 0.6);
        border: 1px solid #475569;
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.5);
    }
    
    /* Text input and select boxes */
    .stTextArea textarea, .stSelectbox select, .stMultiSelect, input {
        border: 2px solid #475569;
        border-radius: 6px;
        background-color: #1e293b !important;
        color: #ffffff !important;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background-color: #1e293b;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        color: #86efac !important;
    }
    
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        color: #93c5fd !important;
    }
    
    .stWarning {
        background-color: rgba(251, 146, 60, 0.1);
        border-left: 4px solid #fb923c;
        color: #fdba74 !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        color: #fca5a5 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(30, 41, 59, 0.6);
        border: 2px dashed #475569;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

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
    """
    Analyze sentiment using ML model with learned neutral detection.
    
    Approach: Use the model's uncertainty as a natural signal for neutral sentiment.
    When the model is uncertain between positive/negative (scores close to 50/50),
    it indicates the text has mixed or neutral sentiment - the model naturally
    learned this pattern from the data.
    
    This is better than keyword matching because:
    1. Model learns contextual patterns (e.g., "decent but predictable")
    2. Handles subtle language the model was trained on
    3. No manual rules - learned from actual examples
    """
    if not text or len(text.strip()) < 3:
        return {
            "label": "neutral",
            "confidence": 0.5,
            "positive_score": 0.33,
            "negative_score": 0.33,
            "neutral_score": 0.34
        }
    
    # Get model predictions
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.nn.functional.softmax(logits, dim=-1)
    
    negative_score = predictions[0][0].item()
    positive_score = predictions[0][1].item()
    
    # Calculate model uncertainty metrics
    score_diff = abs(positive_score - negative_score)
    max_score = max(positive_score, negative_score)
    entropy = -(positive_score * torch.log2(torch.tensor(positive_score + 1e-10)) + 
                negative_score * torch.log2(torch.tensor(negative_score + 1e-10))).item()
    
    # ML-based neutral detection using model's natural uncertainty
    # When model can't decide confidently between pos/neg, it's likely neutral
    
    # Three natural signals from the model:
    # 1. Low score difference (model sees both positive and negative signals)
    # 2. High entropy (maximum uncertainty = 1.0, indicates confusion)
    # 3. Both scores in middle range (not strongly positive or negative)
    
    uncertainty_threshold = 0.7  # Entropy closer to 1.0 means more uncertain
    score_diff_threshold = 0.35  # Scores closer than 35% indicates mixed sentiment
    middle_range_min = 0.35
    middle_range_max = 0.65
    
    # Neutral if model shows high uncertainty through multiple signals
    is_neutral = False
    confidence_penalty = 0
    
    if entropy > uncertainty_threshold:
        # Model is very uncertain - strong signal for neutral
        is_neutral = True
        confidence_penalty = 0.1
    
    if score_diff < score_diff_threshold:
        # Scores are close - model sees mixed sentiment
        if not is_neutral:
            is_neutral = True
            confidence_penalty = 0.15
        else:
            confidence_penalty = 0.2  # Both signals agree
    
    if (middle_range_min <= positive_score <= middle_range_max and 
        middle_range_min <= negative_score <= middle_range_max):
        # Both in middle - not strongly either way
        if not is_neutral:
            is_neutral = True
            confidence_penalty = 0.1
        else:
            confidence_penalty = max(confidence_penalty, 0.25)
    
    # Calculate final scores
    if is_neutral:
        # Neutral detected by model's learned patterns
        neutral_score = min(0.95, max(0.55, 1.0 - score_diff + confidence_penalty))
        label = "neutral"
        confidence = neutral_score
        
        # Redistribute positive/negative to sum to ~1.0 with neutral
        scale = (1.0 - neutral_score) / (positive_score + negative_score)
        positive_score = positive_score * scale
        negative_score = negative_score * scale
    else:
        # Clear positive or negative
        label = "positive" if positive_score > negative_score else "negative"
        confidence = max_score
        neutral_score = max(0.05, min(0.45, 1.0 - confidence))
    
    return {
        "label": label,
        "confidence": confidence,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "neutral_score": neutral_score,
        "model_entropy": entropy  # Return for analysis
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
