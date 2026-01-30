"""
Diabetes Prediction ML Application
Main entry point for the Streamlit app.
"""
import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Diabetes ML Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* ========== MAIN BACKGROUND COLOR ========== */
    /* Darker blue background for main container */
    /* Change 'background-color' to adjust the overall page background */
    .main {
        background-color: #34495e;  /* Main page background - currently dark blue-grey */
    }
    
    /* ========== MAIN TITLE BOX ========== */
    /* Controls the "Diabetes ML Prediction System" header box */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1a5490;  /* Title text color - currently dark blue */
        text-align: center;
        margin-bottom: 1rem;
        background-color: #5dade2;  /* Title box background - currently medium blue */
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* ========== SUBTITLE TEXT ========== */
    /* Controls "Leveraging Machine Learning..." subtitle */
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;  /* Subtitle text color - currently dark grey */
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* ========== INFO BOXES ========== */
    /* Controls information boxes throughout the app */
    .info-box {
        background-color: #ebf5fb;  /* Info box background - currently very light blue */
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #5dade2;  /* Left border accent color - currently medium blue */
        margin: 1rem 0;
        color: #1c2833;  /* Text color inside info boxes - currently very dark grey */
    }
    
    /* ========== FEATURE CARDS ========== */
    /* Controls the three feature boxes (Prediction, Risk, Classification) */
    .feature-card {
        background-color: white;  /* Card background - currently white */
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(93, 173, 226, 0.3);
        margin: 1rem 0;
        transition: transform 0.3s ease;
        border: 2px solid #5dade2;  /* Card border color - currently medium blue */
        min-height: 200px;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(93, 173, 226, 0.5);
    }
    .feature-card h3 {
        color: #1a5490;  /* Feature card heading color - currently dark blue */
    }
    .feature-card p {
        color: #34495e;  /* Feature card text color - currently dark grey */
    }
    
    /* ========== BUTTONS ========== */
    /* Controls all buttons in the application */
    .stButton>button {
        width: 100%;
        background-color: #5dade2;  /* Button background - currently medium blue */
        color: white;  /* Button text color - currently white */
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    
    /* ========== GENERAL TEXT ========== */
    /* Fix markdown text color in various containers */
    .element-container p {
        color: white;  /* General paragraph text - currently dark grey */
    }
    </style>
""", unsafe_allow_html=True)

# Main page content
st.markdown('<h1 class="main-header">🩺 Diabetes ML Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Leveraging Machine Learning to Understand and Predict Diabetes</p>', unsafe_allow_html=True)

# Introduction
with st.container():
    #st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 👋 Welcome to the Diabetes ML Prediction System
    
    This application demonstrates the practical application of various machine learning techniques 
    to predict diabetes, assess risk levels, and classify diabetes types. Built with a focus on 
    **interpretability** and **user experience**, this system combines both supervised and unsupervised 
    learning approaches.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Project Overview
st.markdown("---")
st.header("📊 What This Application Does")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Diabetes Prediction")
    st.write("""
    Uses **Supervised Learning** models (Naive Bayes, Random Forest, Logistic Regression) 
    to predict whether a patient has diabetes based on key health indicators.
    """)

with col2:
    st.markdown("### ⚠️ Risk Assessment")
    st.write("""
    Employs **K-Means Clustering** to group non-diabetic patients into risk categories, 
    helping identify individuals who may be at higher risk for developing diabetes.
    """)

with col3:
    st.markdown("### 🔬 Type Classification")
    st.write("""
    Uses **PCA + SVM** to classify diabetic patients into Type 1 or Type 2 diabetes 
    based on their health profile characteristics.
    """)

# Dataset Information
st.markdown("---")
st.header("📁 About the Dataset")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    The application uses a **synthetically generated diabetes dataset** containing patient medical information.
    
    > 💡 **Note**: When you see "Generate Data" in the application, it means creating synthetic (computer-generated) 
    patient data based on realistic medical parameters. This allows for demonstration and learning without using 
    real patient information.
    
    **Dataset Features Include:**
    
    - **Glucose Levels**: Blood glucose concentration
    - **BMI (Body Mass Index)**: Weight to height ratio
    - **Insulin**: Serum insulin levels
    - **Blood Pressure**: Diastolic blood pressure
    - **Age**: Patient age
    - **Symptoms**: Fatigue, frequent urination, excessive thirst
    - **Lifestyle Factors**: Physical activity level, stress level
    - **Family History**: Genetic predisposition to diabetes
    """)

with col2:
    st.info("""
    **Dataset Stats**
    
    - 500-2,000 patients (configurable)
    - 13 features
    - Balanced classes
    - Synthetically generated
    - Based on medical research
    """)

# Key Features
st.markdown("---")
st.header("🚀 Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Machine Learning Techniques")
    st.markdown("""
    - ✅ **Feature Selection**: Correlation analysis to identify key predictors
    - ✅ **Model Comparison**: Multiple algorithms evaluated side-by-side
    - ✅ **Hyperparameter Tuning**: Grid search for optimal performance
    - ✅ **Dimensionality Reduction**: PCA for complex pattern discovery
    """)

with col2:
    st.markdown("### Interactive Experience")
    st.markdown("""
    - 🎨 **Visual Analytics**: Interactive charts and graphs
    - 🔍 **Real-time Predictions**: Instant diagnosis on custom inputs
    - 📈 **Model Explanations**: Understand how each model works
    - 💡 **Educational Content**: Learn ML concepts through application
    """)

# Navigation Guide
st.markdown("---")
st.header("🧭 Navigation Guide")

st.markdown("""
Use the **sidebar** to navigate between different sections:

1. **🏥 Diabetes Prediction** - Predict diabetes status using different ML models
2. **⚠️ Risk Assessment** - Assess diabetes risk for non-diabetic individuals
3. **🔬 Type Classification** - Classify diabetes into Type 1 or Type 2
4. **📚 Model Explanations** - Learn about the ML models and their applications

Each page provides interactive tools and visualizations to help you understand how 
machine learning can be applied to healthcare diagnostics.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Built with Streamlit</strong> | Using scikit-learn, pandas, and plotly</p>
    <p>This is an educational application demonstrating ML techniques for healthcare analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=80)
    st.title("Navigation")
    st.info("""
    👈 Select a page from the dropdown menu to explore different ML applications.
    
    Each section is designed to be interactive and educational!
    """)
    
    st.markdown("---")
    
    st.markdown("### Quick Stats")
    st.metric("Models Used", "6+")
    st.metric("ML Techniques", "5")
    st.metric("Dataset Size", "1,000")
