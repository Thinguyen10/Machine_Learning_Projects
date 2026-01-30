"""
Diabetes Type Classification Page
PCA + SVM for Type 1 vs Type 2 classification.
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="Type Classification", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner=False)
def train_type_classifier(dataset_size, variance_threshold):
    """Train type classifier with caching."""
    import pandas as pd
    from data_processing import DataGenerator, DataPreprocessor, get_diabetic_data
    from models import DiabetesTypeClassifier
    
    # Generate data
    generator = DataGenerator(n_samples=dataset_size)
    df = generator.generate_data()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df.copy())
    df_encoded = preprocessor.encode_features(df_clean)
    
    # Get diabetic patients only
    diabetic_df = get_diabetic_data(df_encoded)
    
    # Prepare features (drop target)
    X_diabetic = diabetic_df.drop(columns=['diabetes status'])
    
    # Apply PCA and train classifier
    classifier = DiabetesTypeClassifier(variance_threshold=variance_threshold)
    X_pca, cumulative_variance = classifier.apply_pca(X_diabetic)
    cluster_centers = classifier.train(X_diabetic)
    
    # Make predictions on the entire diabetic dataset
    diabetes_types = classifier.predict(X_diabetic)
    diabetic_df['diabetes_type'] = diabetes_types
    
    return df, df_encoded, diabetic_df, X_diabetic, classifier, X_pca, cumulative_variance, cluster_centers, diabetes_types, preprocessor

st.title("🔬 Diabetes Type Classification")
st.markdown("""
Classify diabetic patients into **Type 1** or **Type 2** diabetes using dimensionality reduction (PCA) 
and machine learning (SVM). Since the dataset doesn't include actual type labels, we use clustering 
to create pseudo-labels based on patient characteristics.
""")

# Initialize session state
if 'type_data_loaded' not in st.session_state:
    st.session_state.type_data_loaded = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    dataset_size = st.slider("Dataset Size", 500, 2000, 1000, step=100)
    variance_threshold = st.slider("PCA Variance Threshold", 0.80, 0.99, 0.90, step=0.01,
                                   help="Amount of variance to retain in PCA")
    
    st.markdown("---")
    st.info("""
    **Process Overview:**
    
    1. Filter diabetic patients
    2. Apply PCA for dimensionality reduction
    3. Cluster into 2 groups (Type 1 & Type 2)
    4. Train SVM classifier
    """)
    
    if st.button("🔄 Load New Data", type="primary"):
        st.session_state.type_data_loaded = False

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Type Classification", "👤 Classify Patient", "📚 Understanding PCA"])

# Tab 1: Type Classification
with tab1:
    st.header("Train Type Classifier")
    
    if st.button("▶️ Start Classification", type="primary"):
        with st.spinner("Training classifier..."):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Training type classifier...")
            df, df_encoded, diabetic_df, X_diabetic, classifier, X_pca, cumulative_variance, cluster_centers, diabetes_types, preprocessor = train_type_classifier(
                dataset_size, variance_threshold
            )
            progress_bar.progress(100)
            
            # Store results
            st.session_state.diabetic_df = diabetic_df
            st.session_state.X_diabetic = X_diabetic
            st.session_state.classifier = classifier
            st.session_state.cumulative_variance = cumulative_variance
            st.session_state.cluster_centers = cluster_centers
            st.session_state.diabetes_types = diabetes_types
            st.session_state.type_data_loaded = True
            
            status_text.text("✅ Classification complete!")
            st.balloons()
    
    # Display results
    if st.session_state.type_data_loaded:
        st.success("✅ Type classification completed!")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        type1_count = st.session_state.diabetes_types.count("Type 1 Diabetes")
        type2_count = st.session_state.diabetes_types.count("Type 2 Diabetes")
        
        with col1:
            st.metric("Total Diabetic Patients", len(st.session_state.diabetic_df))
        
        with col2:
            st.metric("Type 1 Diabetes", type1_count)
        
        with col3:
            st.metric("Type 2 Diabetes", type2_count)
        
        with col4:
            st.metric("PCA Components", st.session_state.classifier.n_components)
        
        # PCA Variance Plot
        st.markdown("---")
        st.subheader("📈 PCA - Explained Variance")
        
        fig = plot_pca_variance(
            st.session_state.cumulative_variance,
            st.session_state.classifier.n_components
        )
        st.pyplot(fig)
        
        variance_explained = st.session_state.cumulative_variance[st.session_state.classifier.n_components - 1]
        
        st.info(f"""
        **PCA reduced {st.session_state.X_diabetic.shape[1]} features down to {st.session_state.classifier.n_components} principal components**
        
        These components capture **{variance_explained:.1%}** of the total variance in the data, 
        while significantly reducing complexity and noise.
        """)
        
        # Cluster Analysis
        st.markdown("---")
        st.subheader("🔍 Cluster Center Analysis")
        
        st.markdown("""
        The classifier uses the first principal component (PC1) to distinguish between diabetes types:
        - **Lower PC1 values** → Associated with Type 1 (typically younger, lower BMI)
        - **Higher PC1 values** → Associated with Type 2 (typically older, higher BMI)
        """)
        
        centers = st.session_state.cluster_centers
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Type 1 Cluster")
            st.metric("PC1 Value", f"{centers[0][0]:.2f}")
            st.info("""
            **Typical Characteristics:**
            - Younger age
            - Lower BMI
            - Auto-immune etiology
            - Insulin dependent
            """)
        
        with col2:
            st.markdown("### Type 2 Cluster")
            st.metric("PC1 Value", f"{centers[1][0]:.2f}")
            st.warning("""
            **Typical Characteristics:**
            - Older age
            - Higher BMI
            - Insulin resistance
            - Lifestyle related
            """)
        
        # Type Distribution
        st.markdown("---")
        st.subheader("📊 Diabetes Type Distribution")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            type1_pct = (type1_count / len(st.session_state.diabetes_types)) * 100
            type2_pct = (type2_count / len(st.session_state.diabetes_types)) * 100
            
            distribution_data = pd.DataFrame({
                'Type': ['Type 1', 'Type 2'],
                'Count': [type1_count, type2_count],
                'Percentage': [f"{type1_pct:.1f}%", f"{type2_pct:.1f}%"]
            })
            
            st.dataframe(distribution_data, use_container_width=True)
        
        with col2:
            st.markdown("### Key Differences")
            st.markdown("""
            **Type 1 Diabetes:**
            - Usually diagnosed in childhood/young adults
            - Pancreas produces little/no insulin
            - Requires insulin therapy
            - ~5-10% of all diabetes cases
            
            **Type 2 Diabetes:**
            - Usually diagnosed in adults
            - Body doesn't use insulin properly
            - Often managed with lifestyle changes
            - ~90-95% of all diabetes cases
            """)

# Tab 2: Classify Patient
with tab2:
    st.header("Classify Individual Patient")
    
    if not st.session_state.type_data_loaded:
        st.warning("⚠️ Please train the classifier first in the 'Type Classification' tab.")
    else:
        st.markdown("Enter patient information to classify their diabetes type:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics & Metrics")
            
            age = st.slider("Age", 18, 80, 40)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            bmi = st.slider("BMI", 15.0, 50.0, 28.0, step=0.1)
            
        with col2:
            st.subheader("Lab Values")
            
            glucose = st.slider("Glucose Level (mg/dL)", 100, 300, 180)
            insulin = st.slider("Insulin Level (μU/mL)", 50, 300, 150)
            bp = st.slider("Blood Pressure (mm Hg)", 60, 140, 90)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Lifestyle")
            
            activity = st.selectbox("Physical Activity", [0, 1, 2], 
                                   format_func=lambda x: ["Low", "Medium", "High"][x])
            stress = st.slider("Stress Level", 1, 10, 5)
        
        with col4:
            st.subheader("History & Symptoms")
            
            family_history = st.selectbox("Family History", [0, 1], 
                                         format_func=lambda x: "No" if x == 0 else "Yes")
            fatigue = st.selectbox("Fatigue", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            urination = st.selectbox("Frequent Urination", [0, 1], 
                                    format_func=lambda x: "No" if x == 0 else "Yes")
            thirst = st.selectbox("Excessive Thirst", [0, 1], 
                                 format_func=lambda x: "No" if x == 0 else "Yes")
        
        if st.button("🔬 Classify Diabetes Type", type="primary"):
            # Create patient data
            patient_data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'bmi': [bmi],
                'glucose': [glucose],
                'blood pressure': [bp],
                'insulin': [insulin],
                'physical activity level': [activity],
                'stress level': [stress],
                'family history': [family_history],
                'fatigue': [fatigue],
                'frequent urination': [urination],
                'thirst': [thirst]
            })
            
            # Classify
            diabetes_type = st.session_state.classifier.predict(patient_data)[0]
            
            st.markdown("---")
            st.subheader("🩺 Classification Result")
            
            if diabetes_type == "Type 1 Diabetes":
                st.error(f"""
                ### 🔴 {diabetes_type}
                
                Based on the patient's health profile, the classifier predicts **Type 1 Diabetes**.
                
                **Typical Management:**
                - Daily insulin injections or insulin pump
                - Frequent blood glucose monitoring
                - Carbohydrate counting
                - Regular endocrinologist visits
                
                **Important Note:** This is a prediction based on clustering. Clinical diagnosis 
                should include antibody testing (GAD, IA-2) and C-peptide levels.
                """)
            else:
                st.warning(f"""
                ### 🟡 {diabetes_type}
                
                Based on the patient's health profile, the classifier predicts **Type 2 Diabetes**.
                
                **Typical Management:**
                - Lifestyle modifications (diet, exercise)
                - Oral medications (metformin, etc.)
                - Weight management
                - Regular monitoring
                - Insulin if needed in advanced stages
                
                **Important Note:** This is a prediction based on clustering. Clinical diagnosis 
                should be confirmed with appropriate tests and medical evaluation.
                """)
            
            # Visualize patient profile
            st.markdown("---")
            st.subheader("📊 Patient Health Profile")
            
            patient_values = np.array([age/80, gender, bmi/50, glucose/300, insulin/300, bp/140])
            feature_names = ['Age', 'Gender', 'BMI', 'Glucose', 'Insulin', 'BP']
            
            fig = plot_patient_profile_radar(patient_values, feature_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance explanation
            st.markdown("---")
            st.subheader("📈 Key Factors in Classification")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Factors suggesting Type 1:**
                - Younger age
                - Lower BMI
                - Sudden onset symptoms
                - Strong symptoms despite treatment
                """)
            
            with col2:
                st.markdown("""
                **Factors suggesting Type 2:**
                - Older age
                - Higher BMI
                - Gradual onset
                - Family history of Type 2
                """)

# Tab 3: Understanding PCA
with tab3:
    st.header("Understanding PCA (Principal Component Analysis)")
    
    st.markdown("""
    ### What is PCA?
    
    **Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms 
    high-dimensional data into a smaller set of uncorrelated variables called principal components.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Why Use PCA?
        
        1. **Reduce Complexity**
           - Simplify models with many features
           - Remove redundant information
           - Focus on what matters most
        
        2. **Remove Noise**
           - Filter out random variations
           - Keep only signal
           - Improve model accuracy
        
        3. **Visualize Data**
           - Project high-dimensional data to 2D/3D
           - Identify clusters and patterns
           - Understand data structure
        
        4. **Improve Performance**
           - Faster training times
           - Reduced overfitting
           - Better generalization
        """)
    
    with col2:
        st.markdown("""
        ### How PCA Works
        
        **Step 1: Standardize**
        - Scale all features to same range
        - Prevent features with large values from dominating
        
        **Step 2: Compute Covariance**
        - Find relationships between features
        - Identify correlated variables
        
        **Step 3: Find Principal Components**
        - Calculate eigenvectors and eigenvalues
        - Eigenvectors = directions of maximum variance
        - Eigenvalues = amount of variance in each direction
        
        **Step 4: Select Components**
        - Keep components that explain most variance
        - Typically aim for 90-95% variance explained
        - Discard remaining components (noise)
        """)
    
    st.markdown("---")
    st.subheader("🔍 PCA in Diabetes Type Classification")
    
    st.markdown("""
    In this application, we use PCA to:
    
    1. **Reduce 12 patient features** (age, BMI, glucose, etc.) to just a few principal components
    2. **Capture the most important patterns** that distinguish diabetes types
    3. **Remove noise and correlations** that could confuse the classifier
    4. **Enable clear separation** between Type 1 and Type 2 characteristics
    
    The first principal component (PC1) often captures the age-BMI relationship, which is 
    a key distinguishing factor between Type 1 (younger, lower BMI) and Type 2 (older, higher BMI).
    """)
    
    if st.session_state.type_data_loaded:
        st.markdown("---")
        st.subheader("📊 Your Model's PCA Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Features", st.session_state.X_diabetic.shape[1])
        
        with col2:
            st.metric("Principal Components", st.session_state.classifier.n_components)
        
        with col3:
            variance = st.session_state.cumulative_variance[st.session_state.classifier.n_components - 1]
            st.metric("Variance Retained", f"{variance:.1%}")
        
        reduction_pct = (1 - st.session_state.classifier.n_components / st.session_state.X_diabetic.shape[1]) * 100
        
        st.success(f"""
        **Dimensionality reduced by {reduction_pct:.1f}%** while retaining {variance:.1%} of the information!
        
        This means we've simplified the model significantly while keeping almost all the 
        useful information for classification.
        """)
    
    st.markdown("---")
    st.subheader("⚖️ Trade-offs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Advantages
        - Reduces overfitting
        - Faster computations
        - Removes multicollinearity
        - Visualizable in lower dimensions
        - Works well with many features
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Limitations
        - Loss of interpretability
        - Linear transformation only
        - Variance ≠ importance
        - Sensitive to scaling
        - Requires choosing number of components
        """)

# Additional information
st.markdown("---")
with st.expander("ℹ️ About This Classification Method"):
    st.markdown("""
    ### Methodology
    
    Since our dataset doesn't include actual Type 1/Type 2 labels, we use an **unsupervised learning approach**:
    
    1. **PCA** reduces features to principal components that capture the most variance
    2. **K-Means clustering** (k=2) groups patients into two clusters based on their PC values
    3. **Cluster interpretation**: Lower PC1 → Type 1, Higher PC1 → Type 2
    4. **SVM** is trained on these pseudo-labels for future classification
    
    ### Clinical Reality
    
    In real medical practice, diabetes type is determined through:
    - **Antibody tests** (GAD, IA-2, ZnT8) - positive in Type 1
    - **C-peptide levels** - low in Type 1, normal/high in Type 2
    - **Clinical presentation** - age, onset speed, family history
    - **Response to treatment**
    
    ### Disclaimer
    
    This tool is for **educational purposes** only. It demonstrates ML techniques but should 
    **never replace proper medical diagnosis**. Always consult healthcare professionals for 
    accurate diabetes type determination and treatment planning.
    """)
