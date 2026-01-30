"""
Risk Assessment Page
K-Means clustering for non-diabetic patient risk grouping.
"""
import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import data processing and models
from data_processing import DataGenerator, DataPreprocessor, get_non_diabetic_data
from models import RiskClusterer

# Import visualization functions
from visualizations import plot_elbow_curve, plot_patient_profile_radar, plot_risk_distribution

st.set_page_config(page_title="Risk Assessment", page_icon="⚠️", layout="wide")

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


@st.cache_data(show_spinner=False)
def generate_and_cluster_risk_data(dataset_size):
    """Generate data and perform risk clustering with caching."""
    # Generate data
    generator = DataGenerator(n_samples=dataset_size)
    df = generator.generate_data()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df.copy())
    df_encoded = preprocessor.encode_features(df_clean)
    
    # Get non-diabetic patients
    non_diabetic_df = get_non_diabetic_data(df_encoded)
    
    # Get top features
    top_features = preprocessor.get_top_features(df_encoded)
    
    # Prepare clustering data
    X_cluster = non_diabetic_df[top_features].copy()
    X_cluster = X_cluster.drop(columns=['diabetes status'], errors='ignore')
    
    # Find optimal K
    clusterer = RiskClusterer()
    optimal_k, sse_values = clusterer.find_optimal_k(X_cluster)
    
    # Perform clustering
    clusters = clusterer.fit(X_cluster)
    
    return df, df_encoded, non_diabetic_df, X_cluster, clusterer, optimal_k, sse_values, clusters, top_features, preprocessor

st.title("⚠️ Diabetes Risk Assessment")
st.markdown("""
Identify individuals at **higher risk** of developing diabetes using unsupervised learning.
This tool uses **K-Means clustering** to group non-diabetic patients by their health profiles
and identify those who may need preventive interventions.
""")

# Initialize session state
if 'risk_data_loaded' not in st.session_state:
    st.session_state.risk_data_loaded = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    dataset_size = st.slider("Dataset Size", 500, 2000, 1000, step=100)
    
    st.markdown("---")
    st.info("""
    **How it works:**
    
    1. Generate patient data
    2. Filter non-diabetic patients
    3. Cluster by health features
    4. Identify high-risk groups
    """)
    
    if st.button("🔄 Load New Data", type="primary"):
        st.session_state.risk_data_loaded = False

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Risk Clustering", "👤 Individual Assessment", "📊 Risk Analysis"])

# Tab 1: Risk Clustering
with tab1:
    st.header("Cluster Non-Diabetic Patients by Risk")
    
    if st.button("▶️ Start Risk Assessment", type="primary"):
        with st.spinner("Loading data and performing clustering..."):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Performing risk assessment...")
            df, df_encoded, non_diabetic_df, X_cluster, clusterer, optimal_k, sse_values, clusters, top_features, preprocessor = generate_and_cluster_risk_data(dataset_size)
            
            risk_labels = clusterer.predict_risk(X_cluster)
            progress_bar.progress(100)
            
            # Store results
            non_diabetic_df['risk_level'] = risk_labels
            
            st.session_state.non_diabetic_df = non_diabetic_df
            st.session_state.X_cluster = X_cluster
            st.session_state.clusterer = clusterer
            st.session_state.optimal_k = optimal_k
            st.session_state.sse_values = sse_values
            st.session_state.risk_labels = risk_labels
            st.session_state.top_features = top_features
            st.session_state.risk_data_loaded = True
            
            status_text.text("✅ Risk assessment complete!")
            st.balloons()
    
    # Display results
    if st.session_state.risk_data_loaded:
        st.success("✅ Risk clustering completed!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Non-Diabetic Patients", len(st.session_state.non_diabetic_df))
        
        with col2:
            high_risk_count = st.session_state.risk_labels.count('High Risk')
            st.metric("High Risk Patients", high_risk_count)
        
        with col3:
            low_risk_count = st.session_state.risk_labels.count('Low Risk')
            st.metric("Low Risk Patients", low_risk_count)
        
        # Elbow curve
        st.markdown("---")
        st.subheader("📈 Elbow Method - Optimal Cluster Selection")
        
        fig = plot_elbow_curve(
            range(1, 11),
            st.session_state.sse_values,
            st.session_state.optimal_k,
            title='Elbow Method for Risk Clustering'
        )
        st.pyplot(fig)
        
        st.info(f"""
        **Optimal number of clusters: {st.session_state.optimal_k}**
        
        The elbow point represents the ideal number of clusters where adding more clusters 
        doesn't significantly improve the grouping. This helps us identify distinct risk groups.
        """)
        
        # Risk distribution
        st.markdown("---")
        st.subheader("🥧 Risk Distribution")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = plot_risk_distribution(st.session_state.risk_labels)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Risk Level Breakdown")
            
            high_risk_pct = (high_risk_count / len(st.session_state.risk_labels)) * 100
            low_risk_pct = (low_risk_count / len(st.session_state.risk_labels)) * 100
            
            st.metric("High Risk Percentage", f"{high_risk_pct:.1f}%")
            st.metric("Low Risk Percentage", f"{low_risk_pct:.1f}%")
            
            st.warning(f"""
            **{high_risk_count}** patients are classified as high risk and may benefit from:
            - Regular health monitoring
            - Lifestyle interventions
            - Preventive care programs
            """)
        
        # Cluster centroids
        st.markdown("---")
        st.subheader("📊 Cluster Characteristics")
        
        st.markdown("""
        These are the average feature values for each cluster. The cluster with higher values
        (especially in glucose, insulin, BMI) is identified as the high-risk group.
        """)
        
        centroids_df = st.session_state.clusterer.centroids.copy()
        centroids_df['Risk Level'] = [
            'High Risk' if i == st.session_state.clusterer.high_risk_cluster else 'Low Risk'
            for i in range(len(centroids_df))
        ]
        
        st.dataframe(
            centroids_df.style.highlight_max(axis=0, color='lightcoral'),
            use_container_width=True
        )

# Tab 2: Individual Assessment
with tab2:
    st.header("Assess Individual Risk")
    
    if not st.session_state.risk_data_loaded:
        st.warning("⚠️ Please perform risk clustering first in the 'Risk Clustering' tab.")
    else:
        st.markdown("Enter patient information to assess their diabetes risk:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Health Metrics")
            
            glucose = st.slider("Glucose Level (mg/dL)", 50, 200, 100)
            insulin = st.slider("Insulin Level (μU/mL)", 0, 250, 80)
            bmi = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)
            bp = st.slider("Blood Pressure (mm Hg)", 40, 120, 75)
        
        with col2:
            st.subheader("Symptoms")
            
            thirst = st.selectbox("Excessive Thirst", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            urination = st.selectbox("Frequent Urination", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            fatigue = st.selectbox("Fatigue", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        if st.button("🔍 Assess Risk", type="primary"):
            # Create patient data
            patient_data = pd.DataFrame({
                'glucose': [glucose],
                'thirst': [thirst],
                'insulin': [insulin],
                'frequent urination': [urination],
                'blood pressure': [bp],
                'bmi': [bmi],
                'fatigue': [fatigue]
            })
            
            # Predict risk
            risk_prediction = st.session_state.clusterer.predict_risk(patient_data)[0]
            
            st.markdown("---")
            st.subheader("🩺 Risk Assessment Result")
            
            if risk_prediction == "High Risk":
                st.error("""
                ### 🔴 HIGH RISK
                
                This patient shows characteristics similar to individuals at higher risk of developing diabetes.
                
                **Recommendations:**
                - Schedule regular health check-ups
                - Implement lifestyle modifications (diet, exercise)
                - Monitor glucose levels regularly
                - Consider consulting with a healthcare provider
                """)
            else:
                st.success("""
                ### 🟢 LOW RISK
                
                This patient's health profile suggests a lower risk of developing diabetes.
                
                **Recommendations:**
                - Maintain current healthy lifestyle
                - Continue regular exercise
                - Annual health screenings
                - Stay informed about diabetes prevention
                """)
            
            # Visualize profile
            st.markdown("---")
            st.subheader("📊 Patient Health Profile")
            
            patient_values = [glucose, thirst, insulin, urination, bp, bmi, fatigue]
            feature_names = ['Glucose', 'Thirst', 'Insulin', 'Urination', 'BP', 'BMI', 'Fatigue']
            
            fig = plot_patient_profile_radar(np.array(patient_values), feature_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare to cluster averages
            st.markdown("---")
            st.subheader("📈 Comparison to Cluster Averages")
            
            centroids = st.session_state.clusterer.centroids
            
            comparison_data = []
            for idx, row in centroids.iterrows():
                risk_level = 'High Risk' if idx == st.session_state.clusterer.high_risk_cluster else 'Low Risk'
                comparison_data.append({
                    'Risk Group': risk_level,
                    'Avg Glucose': f"{row['glucose']:.1f}",
                    'Avg Insulin': f"{row['insulin']:.1f}",
                    'Avg BMI': f"{row['bmi']:.1f}",
                    'Avg BP': f"{row['blood pressure']:.1f}"
                })
            
            # Add patient data
            comparison_data.append({
                'Risk Group': 'Your Patient',
                'Avg Glucose': f"{glucose:.1f}",
                'Avg Insulin': f"{insulin:.1f}",
                'Avg BMI': f"{bmi:.1f}",
                'Avg BP': f"{bp:.1f}"
            })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

# Tab 3: Risk Analysis
with tab3:
    st.header("Risk Factor Analysis")
    
    if not st.session_state.risk_data_loaded:
        st.warning("⚠️ Please perform risk clustering first in the 'Risk Clustering' tab.")
    else:
        st.markdown("""
        Understanding the key differences between high-risk and low-risk groups helps us
        identify the most important factors in diabetes risk.
        """)
        
        # Feature comparison
        st.markdown("---")
        st.subheader("📊 Feature Comparison: High Risk vs Low Risk")
        
        df = st.session_state.non_diabetic_df
        
        # Calculate averages for each risk group
        high_risk_df = df[df['risk_level'] == 'High Risk']
        low_risk_df = df[df['risk_level'] == 'Low Risk']
        
        feature_comparison = []
        key_features = ['glucose', 'insulin', 'bmi', 'blood pressure', 'thirst', 'frequent urination', 'fatigue']
        
        for feature in key_features:
            if feature in df.columns:
                high_avg = high_risk_df[feature].mean()
                low_avg = low_risk_df[feature].mean()
                diff = ((high_avg - low_avg) / low_avg) * 100 if low_avg != 0 else 0
                
                feature_comparison.append({
                    'Feature': feature.title(),
                    'High Risk Avg': f"{high_avg:.2f}",
                    'Low Risk Avg': f"{low_avg:.2f}",
                    'Difference (%)': f"{diff:+.1f}%"
                })
        
        comparison_df = pd.DataFrame(feature_comparison)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### High-Risk Characteristics
            
            Patients in the high-risk group typically show:
            - **Higher glucose levels** - Approaching pre-diabetic range
            - **Elevated insulin** - Possible insulin resistance
            - **Higher BMI** - Increased body weight
            - **More symptoms** - Fatigue, thirst, urination
            
            These factors combined suggest metabolic changes that could
            lead to diabetes if not addressed.
            """)
        
        with col2:
            st.markdown("""
            ### Preventive Actions
            
            For high-risk individuals, consider:
            - **Diet modification** - Reduce sugar and refined carbs
            - **Regular exercise** - 30+ minutes daily
            - **Weight management** - Aim for healthy BMI
            - **Stress reduction** - Better sleep and stress control
            - **Regular monitoring** - Check glucose quarterly
            
            Early intervention can significantly reduce the risk of
            developing Type 2 diabetes.
            """)
        
        # Sample patients from each group
        st.markdown("---")
        st.subheader("👥 Sample Patients by Risk Group")
        
        risk_group = st.radio("Select risk group to view:", ["High Risk", "Low Risk"])
        
        sample_df = df[df['risk_level'] == risk_group].head(10)
        st.dataframe(sample_df[key_features + ['risk_level']], use_container_width=True)
        
        # Download option
        st.markdown("---")
        st.download_button(
            label="📥 Download Full Risk Assessment Report",
            data=df.to_csv(index=False),
            file_name="risk_assessment_report.csv",
            mime="text/csv"
        )

# Additional info
st.markdown("---")
with st.expander("ℹ️ About K-Means Clustering for Risk Assessment"):
    st.markdown("""
    ### How K-Means Clustering Works
    
    K-Means is an unsupervised learning algorithm that groups similar data points together.
    
    **Process:**
    1. **Initialize** K cluster centers randomly
    2. **Assign** each patient to the nearest cluster center
    3. **Update** cluster centers based on assigned patients
    4. **Repeat** until clusters stabilize
    
    **Why it's useful for risk assessment:**
    - Identifies natural groupings in patient data
    - No labeled data required
    - Reveals patterns not obvious in individual features
    - Helps prioritize preventive care resources
    
    **Limitations:**
    - Number of clusters (K) must be chosen
    - Results can vary with initialization
    - Assumes spherical cluster shapes
    - Should be combined with clinical judgment
    """)
