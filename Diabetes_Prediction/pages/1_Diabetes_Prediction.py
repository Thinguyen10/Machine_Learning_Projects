"""
Diabetes Prediction Page
Compare and use different ML models for diabetes prediction.
"""
import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import visualization functions
from visualizations import plot_correlation_matrix, plot_patient_profile_radar, plot_distribution_comparison

st.set_page_config(page_title="Diabetes Prediction", page_icon="🏥", layout="wide")

# Apply consistent styling from main app
st.markdown("""
    <style>
    .main {
        background-color: #34495e;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    .stMarkdown {
        color: white;
    }
    p {
        color: white;
    }
    label {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Cached data generation
@st.cache_data(show_spinner=False)
def generate_and_prepare_data(dataset_size, test_size):
    """Generate and prepare data with caching."""
    import pandas as pd
    from data_processing import DataGenerator, DataPreprocessor
    
    generator = DataGenerator(n_samples=dataset_size)
    df = generator.generate_data()
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, top_features = preprocessor.prepare_data(
        df, test_size=test_size
    )
    
    df_processed = preprocessor.encode_features(preprocessor.clean_data(df.copy()))
    
    return df, df_processed, X_train, X_test, y_train, y_test, top_features, preprocessor


# Cached model training
@st.cache_resource(show_spinner=False)
def train_all_models(_X_train, _y_train, _X_test, _y_test, tune_rf=False):
    """Train all models with caching."""
    from models import NaiveBayesModel, RandomForestModel, LogisticRegressionModel
    
    models_dict = {}
    metrics_dict = {}
    
    # Naive Bayes
    nb_model = NaiveBayesModel()
    nb_model.train(_X_train, _y_train)
    nb_metrics = nb_model.evaluate(_X_test, _y_test)
    models_dict['Naive Bayes'] = nb_model
    metrics_dict['Naive Bayes'] = nb_metrics
    
    # Random Forest
    rf_model = RandomForestModel(tune_hyperparameters=tune_rf)
    rf_model.train(_X_train, _y_train)
    rf_metrics = rf_model.evaluate(_X_test, _y_test)
    models_dict['Random Forest'] = rf_model
    metrics_dict['Random Forest'] = rf_metrics
    
    # Logistic Regression
    lr_model = LogisticRegressionModel()
    lr_model.train(_X_train, _y_train)
    lr_metrics = lr_model.evaluate(_X_test, _y_test)
    models_dict['Logistic Regression'] = lr_model
    metrics_dict['Logistic Regression'] = lr_metrics
    
    return models_dict, metrics_dict

st.title("🏥 Diabetes Status Prediction")
st.markdown("""
Use machine learning models to predict whether a patient has diabetes based on their health indicators.
Compare the performance of **Naive Bayes**, **Random Forest**, and **Logistic Regression**.
""")

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
    st.session_state.models_trained = False

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    dataset_size = st.slider("Dataset Size", 500, 2000, 1000, step=100)
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, step=0.05)
    
    st.markdown("---")
    
    tune_rf = st.checkbox("Tune Random Forest Hyperparameters", value=False,
                         help="Enable grid search for optimal parameters (slower)")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Training", "🔍 Make Predictions", "📈 Data Exploration", "🎯 Model Comparison"])

# Tab 1: Model Training
with tab1:
    st.header("Train and Evaluate Models")
    
    if st.button("▶️ Start Training", type="primary", key="train_btn"):
        with st.spinner("Generating data and training models..."):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Generate data (cached)
            status_text.text("Generating synthetic dataset...")
            df, df_processed, X_train, X_test, y_train, y_test, top_features, preprocessor = generate_and_prepare_data(
                dataset_size, test_size
            )
            progress_bar.progress(40)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.df_processed = df_processed
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.top_features = top_features
            st.session_state.preprocessor = preprocessor
            st.session_state.data_generated = True
            
            # Train models (cached)
            status_text.text("Training models...")
            models_dict, metrics_dict = train_all_models(
                X_train, y_train, X_test, y_test, tune_rf
            )
            progress_bar.progress(100)
            
            # Store models
            st.session_state.models = models_dict
            st.session_state.metrics = metrics_dict
            st.session_state.models_trained = True
            
            status_text.text("✅ Training complete!")
            st.balloons()
    
    # Display results
    if st.session_state.models_trained:
        from visualizations import display_metrics_cards, plot_metrics_comparison
        
        st.success("✅ All models trained successfully!")
        
        st.markdown("---")
        st.subheader("📊 Model Performance Metrics")
        
        # Display metrics for each model
        for model_name, metrics in st.session_state.metrics.items():
            with st.expander(f"📈 {model_name}", expanded=True):
                display_metrics_cards(metrics)
                
                # Classification report
                st.markdown("**Detailed Classification Report:**")
                report_df = pd.DataFrame(metrics['report']).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🎯 Model Comparison")
        fig = plot_metrics_comparison(st.session_state.metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = max(st.session_state.metrics.items(), key=lambda x: x[1]['accuracy'])
        st.success(f"🏆 **Best Performing Model:** {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.2%})")

# Tab 2: Make Predictions
with tab2:
    st.header("Make Predictions on New Patients")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first in the 'Model Training' tab.")
    else:
        st.markdown("Enter patient information to predict diabetes status:")
        
        # Get the features used during training
        top_features = st.session_state.top_features
        df = st.session_state.df
        df_processed = st.session_state.df_processed
        
        # Create input form based on top features
        st.info(f"📋 Using {len(top_features)} most important features: {', '.join(top_features)}")
        
        col1, col2 = st.columns(2)
        
        # Collect input for each feature
        patient_input = {}
        
        with col1:
            st.subheader("Patient Information")
            
            # Loop through first half of selected features to create input widgets
            # Split features across two columns for better UI layout
            for feature in top_features[:len(top_features)//2]:
                # Create appropriate input widget based on feature type
                if feature in ['glucose']:
                    patient_input[feature] = st.slider("Glucose Level (mg/dL)", 50, 250, 120, key=feature)
                elif feature in ['insulin']:
                    patient_input[feature] = st.slider("Insulin Level (μU/mL)", 0, 300, 100, key=feature)
                elif feature in ['bmi']:
                    patient_input[feature] = st.slider("BMI", 15.0, 50.0, 25.0, step=0.1, key=feature)
                elif feature in ['blood pressure']:
                    patient_input[feature] = st.slider("Blood Pressure (mm Hg)", 40, 140, 80, key=feature)
                elif feature in ['age']:
                    patient_input[feature] = st.slider("Age", 18, 80, 45, key=feature)
                elif feature in ['stress level']:
                    patient_input[feature] = st.slider("Stress Level (1-10)", 1, 10, 5, key=feature)
                elif feature in ['gender']:
                    gender_input = st.selectbox("Gender", ["Male", "Female"], key=feature)
                    patient_input[feature] = 1 if gender_input == "Male" else 0  # Encode
                elif feature in ['physical activity level']:
                    activity_input = st.selectbox("Physical Activity", ["Low", "Medium", "High"], key=feature)
                    patient_input[feature] = {"Low": 0, "Medium": 1, "High": 2}[activity_input]
                elif feature in ['family history']:
                    history_input = st.selectbox("Family History", ["No", "Yes"], key=feature)
                    patient_input[feature] = 1 if history_input == "Yes" else 0
                elif feature in ['thirst']:
                    patient_input[feature] = st.selectbox("Excessive Thirst", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=feature)
                elif feature in ['frequent urination']:
                    patient_input[feature] = st.selectbox("Frequent Urination", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=feature)
                elif feature in ['fatigue']:
                    patient_input[feature] = st.selectbox("Fatigue", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=feature)
                else:
                    # Default handler for any feature not explicitly handled above
                    # Uses median value from training data as default to ensure realistic input
                    if feature in df_processed.columns:
                        median_val = df_processed[feature].median()
                        patient_input[feature] = st.number_input(f"{feature.title()}", value=float(median_val), key=feature)
        
        with col2:
            st.subheader("Symptoms & History")
            
            # Loop through second half of selected features
            for feature in top_features[len(top_features)//2:]:
                # Create appropriate input widget based on feature type
                if feature in ['thirst']:
                    patient_input[feature] = st.selectbox("Excessive Thirst", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=feature)
                elif feature in ['frequent urination']:
                    patient_input[feature] = st.selectbox("Frequent Urination", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=feature)
                elif feature in ['fatigue']:
                    patient_input[feature] = st.selectbox("Fatigue", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=feature)
                elif feature in ['glucose']:
                    patient_input[feature] = st.slider("Glucose Level (mg/dL)", 50, 250, 120, key=feature)
                elif feature in ['insulin']:
                    patient_input[feature] = st.slider("Insulin Level (μU/mL)", 0, 300, 100, key=feature)
                elif feature in ['bmi']:
                    patient_input[feature] = st.slider("BMI", 15.0, 50.0, 25.0, step=0.1, key=feature)
                elif feature in ['blood pressure']:
                    patient_input[feature] = st.slider("Blood Pressure (mm Hg)", 40, 140, 80, key=feature)
                elif feature in ['age']:
                    patient_input[feature] = st.slider("Age", 18, 80, 45, key=feature)
                elif feature in ['stress level']:
                    patient_input[feature] = st.slider("Stress Level (1-10)", 1, 10, 5, key=feature)
                elif feature in ['gender']:
                    gender_input = st.selectbox("Gender", ["Male", "Female"], key=feature)
                    patient_input[feature] = 1 if gender_input == "Male" else 0
                elif feature in ['physical activity level']:
                    activity_input = st.selectbox("Physical Activity", ["Low", "Medium", "High"], key=feature)
                    patient_input[feature] = {"Low": 0, "Medium": 1, "High": 2}[activity_input]
                elif feature in ['family history']:
                    history_input = st.selectbox("Family History", ["No", "Yes"], key=feature)
                    patient_input[feature] = 1 if history_input == "Yes" else 0
                else:
                    # Default handler for any other feature - use median from training data
                    if feature in df_processed.columns:
                        median_val = df_processed[feature].median()
                        patient_input[feature] = st.number_input(f"{feature.title()}", value=float(median_val), key=feature)
        
        if st.button("🔮 Predict", type="primary"):
            # Create patient data with features in correct order
            patient_data = pd.DataFrame([patient_input])[top_features]
            
            st.markdown("---")
            st.subheader("🩺 Prediction Results")
            
            # Make predictions with all models
            results_col1, results_col2, results_col3 = st.columns(3)
            
            models = st.session_state.models
            
            with results_col1:
                st.markdown("### Naive Bayes")
                nb_pred = models['Naive Bayes'].predict(patient_data)[0]
                if nb_pred == 1:
                    st.error("🔴 **Diabetic**")
                else:
                    st.success("🟢 **Not Diabetic**")
            
            with results_col2:
                st.markdown("### Random Forest")
                rf_pred = models['Random Forest'].predict(patient_data)[0]
                if rf_pred == 1:
                    st.error("🔴 **Diabetic**")
                else:
                    st.success("🟢 **Not Diabetic**")
            
            with results_col3:
                st.markdown("### Logistic Regression")
                lr_pred = models['Logistic Regression'].predict(patient_data)[0]
                if lr_pred == 1:
                    st.error("🔴 **Diabetic**")
                else:
                    st.success("🟢 **Not Diabetic**")
            
            # Consensus
            st.markdown("---")
            predictions = [nb_pred, rf_pred, lr_pred]
            consensus = 1 if sum(predictions) >= 2 else 0
            
            if consensus == 1:
                st.error("### 🚨 Consensus: **DIABETIC** - Please consult a healthcare professional")
            else:
                st.success("### ✅ Consensus: **NOT DIABETIC** - Continue healthy lifestyle")
            
            # Visualize patient profile
            st.markdown("---")
            st.subheader("📊 Patient Health Profile")
            
            # Prepare values for radar chart (using actual patient input)
            patient_values = list(patient_input.values())
            feature_names = list(patient_input.keys())
            
            fig = plot_patient_profile_radar(np.array(patient_values), feature_names)
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Data Exploration
with tab3:
    st.header("Explore the Dataset")
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first in the 'Model Training' tab.")
    else:
        df_processed = st.session_state.df_processed
        
        # Dataset overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", len(df_processed))
        with col2:
            diabetic_count = (df_processed['diabetes status'] == 1).sum()
            st.metric("Diabetic Patients", diabetic_count)
        with col3:
            non_diabetic_count = (df_processed['diabetes status'] == 0).sum()
            st.metric("Non-Diabetic Patients", non_diabetic_count)
        
        # Correlation matrix
        st.markdown("---")
        st.subheader("📈 Feature Correlation Matrix")
        fig = plot_correlation_matrix(df_processed)
        st.pyplot(fig)
        
        st.info("""
        **Interpretation:** Features with higher correlation (darker colors) to 'diabetes status' 
        are more important for prediction. This is why we selected the top correlated features for our models.
        """)
        
        # Feature distributions
        st.markdown("---")
        st.subheader("📊 Feature Distributions")
        
        feature_to_plot = st.selectbox(
            "Select a feature to visualize:",
            ['glucose', 'insulin', 'bmi', 'blood pressure']
        )
        
        fig = plot_distribution_comparison(df_processed, feature_to_plot)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample data
        st.markdown("---")
        st.subheader("👥 Sample Patient Data")
        st.dataframe(df_processed.head(20), use_container_width=True)

# Tab 4: Model Comparison
with tab4:
    st.header("Model Comparison & Selection")
    
    st.markdown("""
    ### Why Compare Multiple Models?
    
    Different machine learning algorithms have different strengths and weaknesses. By comparing multiple models,
    we can:
    
    - **Identify the best performer** for our specific dataset
    - **Understand trade-offs** between accuracy, precision, and recall
    - **Build confidence** through consensus predictions
    - **Learn** which algorithms work best for medical diagnosis
    """)
    
    if st.session_state.models_trained:
        st.markdown("---")
        
        # Model characteristics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🧮 Naive Bayes
            **Strengths:**
            - Fast training and prediction
            - Works well with small datasets
            - Handles probabilities naturally
            
            **Best for:**
            - Quick diagnostics
            - Baseline comparisons
            """)
        
        with col2:
            st.markdown("""
            ### 🌳 Random Forest
            **Strengths:**
            - Handles non-linear relationships
            - Provides feature importance
            - Robust to overfitting
            
            **Best for:**
            - Complex patterns
            - Feature analysis
            """)
        
        with col3:
            st.markdown("""
            ### 📈 Logistic Regression
            **Strengths:**
            - Interpretable coefficients
            - Probability outputs
            - Industry standard
            
            **Best for:**
            - Clinical settings
            - Explainable AI
            """)
        
        st.markdown("---")
        st.subheader("📊 Performance Comparison")
        
        # Summary table
        summary_data = []
        for model_name, metrics in st.session_state.metrics.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.2%}",
                'Precision': f"{metrics['precision']:.2%}",
                'Recall': f"{metrics['recall']:.2%}",
                'F1 Score': f"{metrics['f1']:.2%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        st.success("""
        💡 **Tip:** In medical diagnostics, **recall** (sensitivity) is often more important than precision,
        as we want to minimize false negatives (missing actual diabetes cases).
        """)
