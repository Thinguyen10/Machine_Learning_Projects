"""
Model Explanations Page
Detailed educational content about all ML models used.
"""
import streamlit as st

st.set_page_config(page_title="Model Explanations", page_icon="📚", layout="wide")

st.title("📚 Machine Learning Models Explained")
st.markdown("""
Learn about the machine learning algorithms used in this diabetes prediction system. 
Each model has unique strengths and applications in healthcare diagnostics.
""")

# Sidebar navigation
with st.sidebar:
    st.header("📖 Quick Navigation")
    
    model_section = st.radio(
        "Jump to section:",
        [
            "Overview",
            "Naive Bayes",
            "Random Forest",
            "Logistic Regression",
            "K-Means Clustering",
            "PCA",
            "SVM",
            "Comparison"
        ]
    )

# Overview
if model_section == "Overview":
    st.header("🎯 Overview")
    
    st.markdown("""
    This application uses a combination of **supervised** and **unsupervised** learning techniques 
    to address different aspects of diabetes prediction and analysis.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎓 Supervised Learning
        
        Models that learn from labeled data (knowing the outcome).
        
        **Used for:**
        - Predicting diabetes status (yes/no)
        - Classifying diabetes types (Type 1/2)
        
        **Models:**
        - Naive Bayes
        - Random Forest
        - Logistic Regression
        - Support Vector Machine (SVM)
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Unsupervised Learning
        
        Models that find patterns without labeled data.
        
        **Used for:**
        - Risk grouping (high/low risk)
        - Dimensionality reduction
        - Pattern discovery
        
        **Models:**
        - K-Means Clustering
        - Principal Component Analysis (PCA)
        """)
    
    st.markdown("---")
    st.subheader("📊 Model Application Matrix")
    
    application_data = {
        'Task': [
            'Diabetes Prediction',
            'Risk Assessment',
            'Type Classification',
            'Feature Selection',
            'Dimensionality Reduction'
        ],
        'Model(s) Used': [
            'Naive Bayes, Random Forest, Logistic Regression',
            'K-Means Clustering',
            'SVM',
            'Correlation Analysis',
            'PCA'
        ],
        'Learning Type': [
            'Supervised',
            'Unsupervised',
            'Supervised',
            'Statistical',
            'Unsupervised'
        ]
    }
    
    st.table(application_data)

# Naive Bayes
elif model_section == "Naive Bayes":
    st.header("🧮 Naive Bayes Classifier")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Naive Bayes?
        
        Naive Bayes is a **probabilistic classifier** based on Bayes' theorem. It's called "naive" 
        because it assumes that all features are independent of each other (which isn't always true, 
        but works surprisingly well).
        
        ### How It Works
        
        Given patient features (glucose, BMI, etc.), it calculates:
        
        **P(Diabetes | Features) = P(Features | Diabetes) × P(Diabetes) / P(Features)**
        
        Where:
        - **P(Diabetes | Features)**: Probability of diabetes given the patient's features
        - **P(Features | Diabetes)**: Likelihood of these features in diabetic patients
        - **P(Diabetes)**: Prior probability of diabetes in the population
        - **P(Features)**: Probability of observing these features
        
        The model calculates this for both "Diabetic" and "Not Diabetic" and chooses the higher probability.
        """)
    
    with col2:
        st.info("""
        **Key Characteristics**
        
        ⚡ **Speed:** Very fast
        
        📊 **Data Needs:** Small-medium
        
        🎯 **Accuracy:** Good
        
        🧠 **Interpretability:** High
        
        ⚙️ **Complexity:** Low
        """)
    
    st.markdown("---")
    st.subheader("✅ Strengths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Fast training and prediction** - Ideal for real-time systems
        - **Works well with small datasets** - Doesn't need massive data
        - **Handles missing data** - Can work with incomplete records
        - **Probabilistic outputs** - Gives confidence levels
        - **Good for baseline** - Quick first model to benchmark against
        """)
    
    with col2:
        st.markdown("""
        - **Easy to implement** - Simple mathematical formula
        - **Scales well** - Works with many features
        - **Resistant to irrelevant features** - Ignores noise well
        - **Good with categorical data** - Natural fit for yes/no features
        """)
    
    st.markdown("---")
    st.subheader("⚠️ Limitations")
    
    st.markdown("""
    - **Independence assumption** - Assumes features don't interact (often violated in medical data)
    - **Zero frequency problem** - If a combination wasn't seen in training, it gets zero probability
    - **Sensitive to feature distribution** - Works best with normally distributed features
    - **Can't capture complex relationships** - Linear decision boundaries only
    """)
    
    st.markdown("---")
    st.subheader("🏥 Application in Diabetes Prediction")
    
    st.markdown("""
    Naive Bayes is excellent for diabetes screening because:
    
    1. **Quick decisions** - Can rapidly screen large populations
    2. **Probability outputs** - Gives risk percentages, not just yes/no
    3. **Interpretable** - Doctors can understand the reasoning
    4. **Efficient** - Low computational cost for deployment
    
    **Example:** A patient with high glucose (180 mg/dL) and high BMI (32) gets analyzed:
    - P(High Glucose | Diabetic) = 0.85
    - P(High BMI | Diabetic) = 0.75
    - Combined with prior probability → Final prediction: 92% chance of diabetes
    """)

# Random Forest
elif model_section == "Random Forest":
    st.header("🌳 Random Forest Classifier")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Random Forest?
        
        Random Forest is an **ensemble learning method** that creates multiple decision trees and 
        combines their predictions. It's like asking many experts and taking a vote on the diagnosis.
        
        ### How It Works
        
        1. **Create Multiple Trees**
           - Build many decision trees (e.g., 100 trees)
           - Each tree uses a random subset of features
           - Each tree sees a random subset of training data (bootstrapping)
        
        2. **Make Predictions**
           - Each tree makes its own prediction
           - Final prediction = majority vote
           - For diabetes: if 70/100 trees say "diabetic", predict diabetic
        
        3. **Feature Importance**
           - Track which features most often lead to correct splits
           - Rank features by their contribution to accuracy
        """)
    
    with col2:
        st.info("""
        **Key Characteristics**
        
        ⚡ **Speed:** Medium
        
        📊 **Data Needs:** Medium-large
        
        🎯 **Accuracy:** Very high
        
        🧠 **Interpretability:** Medium
        
        ⚙️ **Complexity:** Medium-high
        """)
    
    st.markdown("---")
    st.subheader("🌲 Decision Tree Example")
    
    st.code("""
    Root: Glucose Level
    ├─ If Glucose ≤ 140
    │  └─ Check BMI
    │     ├─ If BMI ≤ 25 → NOT DIABETIC ✓
    │     └─ If BMI > 25 → Check Age...
    └─ If Glucose > 140
       └─ Check Insulin
          ├─ If Insulin < 100 → DIABETIC ✗
          └─ If Insulin ≥ 100 → Check Family History...
    """, language="text")
    
    st.markdown("---")
    st.subheader("✅ Strengths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **High accuracy** - Often outperforms single models
        - **Handles non-linearity** - Captures complex patterns
        - **Robust to outliers** - Averaging reduces impact of anomalies
        - **Feature importance** - Shows which features matter most
        - **No feature scaling needed** - Works with different units
        """)
    
    with col2:
        st.markdown("""
        - **Handles missing data** - Can route around missing values
        - **Prevents overfitting** - Randomness reduces memorization
        - **Works with mixed data** - Handles categorical + numerical
        - **Parallel processing** - Trees can be built independently
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Hyperparameters")
    
    st.markdown("""
    Random Forest performance can be tuned with several parameters:
    
    | Parameter | What It Does | Impact |
    |-----------|--------------|--------|
    | **n_estimators** | Number of trees | More trees = better accuracy but slower |
    | **max_depth** | Maximum tree depth | Deeper = more complex patterns but risk overfitting |
    | **min_samples_split** | Min samples to split node | Higher = simpler trees, prevents overfitting |
    | **min_samples_leaf** | Min samples in leaf | Higher = smoother decision boundaries |
    
    This app uses **GridSearchCV** to automatically find optimal values!
    """)
    
    st.markdown("---")
    st.subheader("🏥 Application in Diabetes Prediction")
    
    st.markdown("""
    Random Forest excels in diabetes prediction because:
    
    1. **Captures interactions** - Can learn "high glucose + high BMI + family history = very high risk"
    2. **Feature ranking** - Identifies glucose and insulin as top predictors
    3. **Handles real-world data** - Robust to measurement errors and missing values
    4. **No assumptions** - Doesn't require features to be normally distributed
    
    **In this app:** Often achieves 90%+ accuracy on diabetes prediction!
    """)

# Logistic Regression
elif model_section == "Logistic Regression":
    st.header("📈 Logistic Regression")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Logistic Regression?
        
        Despite its name, Logistic Regression is a **classification algorithm**, not regression. 
        It models the probability that an input belongs to a particular class.
        
        ### How It Works
        
        It uses the **logistic (sigmoid) function** to convert a linear combination of features 
        into a probability between 0 and 1:
        
        **P(Diabetic) = 1 / (1 + e^-(β₀ + β₁×glucose + β₂×BMI + ...))**
        
        Where:
        - **β₀**: Intercept (baseline probability)
        - **β₁, β₂, ...**: Coefficients for each feature
        - **e**: Euler's number (≈2.718)
        
        If P(Diabetic) > 0.5 → Predict Diabetic  
        If P(Diabetic) ≤ 0.5 → Predict Not Diabetic
        """)
    
    with col2:
        st.info("""
        **Key Characteristics**
        
        ⚡ **Speed:** Fast
        
        📊 **Data Needs:** Small-large
        
        🎯 **Accuracy:** Good-very good
        
        🧠 **Interpretability:** Very high
        
        ⚙️ **Complexity:** Low
        """)
    
    st.markdown("---")
    st.subheader("📊 The Sigmoid Function")
    
    st.markdown("""
    The sigmoid function creates an S-shaped curve that:
    - Maps any input to a value between 0 and 1
    - Represents probability smoothly
    - Has a clear decision boundary at 0.5
    """)
    
    st.code("""
    When Linear Score = -5  →  Probability = 0.007 (0.7%) → NOT DIABETIC
    When Linear Score = -2  →  Probability = 0.119 (12%)  → NOT DIABETIC
    When Linear Score =  0  →  Probability = 0.500 (50%)  → BOUNDARY
    When Linear Score = +2  →  Probability = 0.881 (88%)  → DIABETIC
    When Linear Score = +5  →  Probability = 0.993 (99%)  → DIABETIC
    """, language="text")
    
    st.markdown("---")
    st.subheader("✅ Strengths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Interpretable coefficients** - Each feature's contribution is clear
        - **Probability outputs** - Not just yes/no, but confidence levels
        - **Computationally efficient** - Fast training and prediction
        - **Industry standard** - Widely used and trusted in healthcare
        - **Good with linear relationships** - Excels when features correlate linearly with outcome
        """)
    
    with col2:
        st.markdown("""
        - **Regularization support** - L1/L2 to prevent overfitting
        - **Works with small data** - Doesn't need huge datasets
        - **Feature scaling helps** - Works better with standardized features
        - **Stable** - Consistent results, not random
        """)
    
    st.markdown("---")
    st.subheader("🔍 Interpreting Coefficients")
    
    st.markdown("""
    Example coefficients for diabetes prediction:
    
    | Feature | Coefficient | Interpretation |
    |---------|-------------|----------------|
    | **Glucose** | +2.5 | Strong positive: Higher glucose → Higher diabetes risk |
    | **BMI** | +1.2 | Moderate positive: Higher BMI → Higher risk |
    | **Insulin** | +0.8 | Positive: Higher insulin levels → Higher risk |
    | **Exercise** | -0.6 | Negative: More exercise → Lower risk |
    | **Age** | +0.3 | Weak positive: Older age → Slightly higher risk |
    
    A unit increase in glucose increases log-odds of diabetes by 2.5!
    """)
    
    st.markdown("---")
    st.subheader("🏥 Application in Diabetes Prediction")
    
    st.markdown("""
    Logistic Regression is ideal for clinical settings because:
    
    1. **Explainable to doctors** - "Each 1 mg/dL increase in glucose raises diabetes risk by X%"
    2. **Probability outputs** - "This patient has 85% probability of diabetes"
    3. **Regulatory friendly** - Transparent models are easier to approve
    4. **Quick deployment** - Can be implemented in simple spreadsheets
    
    **In this app:** Often achieves the highest accuracy with proper feature scaling!
    """)

# K-Means Clustering
elif model_section == "K-Means Clustering":
    st.header("🎯 K-Means Clustering")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is K-Means?
        
        K-Means is an **unsupervised learning algorithm** that groups similar data points into clusters. 
        It doesn't need labeled data - it finds natural groupings on its own.
        
        ### How It Works
        
        1. **Initialize** K cluster centers randomly
        2. **Assign** each patient to the nearest cluster center
        3. **Update** cluster centers to the mean of assigned patients
        4. **Repeat** steps 2-3 until centers stop moving
        
        ### Mathematical Distance
        
        Uses **Euclidean distance** to measure similarity:
        
        **Distance = √[(glucose₁-glucose₂)² + (BMI₁-BMI₂)² + ...]**
        
        Patients with similar health profiles end up in the same cluster.
        """)
    
    with col2:
        st.info("""
        **Key Characteristics**
        
        ⚡ **Speed:** Fast
        
        📊 **Data Needs:** Medium
        
        🎯 **Accuracy:** N/A (unsupervised)
        
        🧠 **Interpretability:** High
        
        ⚙️ **Complexity:** Low
        """)
    
    st.markdown("---")
    st.subheader("🔄 Algorithm Visualization")
    
    st.code("""
    Iteration 1:
    Center 1: [glucose=120, BMI=25]  →  Assigns: Patients 1,3,5,7
    Center 2: [glucose=180, BMI=32]  →  Assigns: Patients 2,4,6,8
    
    Iteration 2:
    Update Center 1: [glucose=115, BMI=24]  (mean of 1,3,5,7)
    Update Center 2: [glucose=185, BMI=33]  (mean of 2,4,6,8)
    
    Iteration 3:
    Centers barely move → CONVERGED ✓
    """, language="text")
    
    st.markdown("---")
    st.subheader("📊 Choosing K: The Elbow Method")
    
    st.markdown("""
    How do we know the right number of clusters (K)?
    
    **Elbow Method:**
    1. Try different values of K (1, 2, 3, ...)
    2. For each K, calculate total within-cluster distance (SSE)
    3. Plot K vs SSE
    4. Look for the "elbow" - where adding more clusters doesn't help much
    
    **In this app:** We automatically detect the elbow point using mathematical optimization!
    """)
    
    st.markdown("---")
    st.subheader("✅ Strengths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **No labeled data needed** - Finds patterns automatically
        - **Fast and scalable** - Works with large datasets
        - **Easy to implement** - Simple algorithm
        - **Interpretable clusters** - Can understand what makes each group unique
        """)
    
    with col2:
        st.markdown("""
        - **Works well with spherical clusters** - Natural groupings
        - **Feature scaling sensitive** - Standardization helps
        - **Deterministic after convergence** - Stable results
        """)
    
    st.markdown("---")
    st.subheader("⚠️ Limitations")
    
    st.markdown("""
    - **Must choose K** - Number of clusters is a hyperparameter
    - **Sensitive to initialization** - Different starting points can give different results
    - **Assumes spherical clusters** - Struggles with elongated or irregular shapes
    - **Sensitive to outliers** - Extreme values can skew cluster centers
    - **Euclidean distance bias** - Features with larger scales dominate
    """)
    
    st.markdown("---")
    st.subheader("🏥 Application in Risk Assessment")
    
    st.markdown("""
    K-Means is used for diabetes risk grouping because:
    
    1. **No labels needed** - We don't know who's "high risk" beforehand
    2. **Natural groupings** - Patients naturally cluster by health profile
    3. **Identify high-risk group** - Cluster with highest average glucose/BMI = high risk
    4. **Preventive care** - Target interventions to high-risk cluster
    
    **Example:** 
    - Cluster 1 (60% of patients): Low glucose, low BMI → Low Risk
    - Cluster 2 (25%): Medium values → Medium Risk
    - Cluster 3 (15%): High glucose, high BMI → High Risk ⚠️
    
    Focus preventive programs on Cluster 3!
    """)

# PCA
elif model_section == "PCA":
    st.header("🔬 Principal Component Analysis (PCA)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is PCA?
        
        PCA is a **dimensionality reduction technique** that transforms many correlated features 
        into fewer uncorrelated "principal components" while preserving most of the information.
        
        ### Why Reduce Dimensions?
        
        Imagine you have 12 features: age, glucose, BMI, insulin, blood pressure, etc.
        
        - Some features are correlated (glucose and insulin often move together)
        - Some features are noisy (random measurement errors)
        - More features = more complexity = harder to visualize and understand
        
        **PCA's solution:** Find a few "super-features" that capture most of the variation.
        """)
    
    with col2:
        st.info("""
        **Key Characteristics**
        
        ⚡ **Speed:** Medium
        
        📊 **Data Needs:** Medium-large
        
        🎯 **Purpose:** Reduce dimensions
        
        🧠 **Interpretability:** Low-medium
        
        ⚙️ **Complexity:** Medium
        """)
    
    st.markdown("---")
    st.subheader("📐 How PCA Works")
    
    st.markdown("""
    ### Step 1: Standardize Features
    
    Scale all features to have mean=0 and standard deviation=1.
    
    **Why?** If glucose ranges 50-300 and age ranges 20-80, glucose would dominate!
    
    ```
    Standardized value = (original - mean) / std_deviation
    ```
    
    ### Step 2: Compute Covariance Matrix
    
    Find how features relate to each other:
    - Do glucose and insulin increase together? (positive covariance)
    - Does exercise reduce BMI? (negative covariance)
    
    ### Step 3: Calculate Eigenvectors & Eigenvalues
    
    - **Eigenvectors** = directions of maximum variance (the principal components)
    - **Eigenvalues** = amount of variance in each direction
    
    Think of it as finding the "natural axes" of your data.
    
    ### Step 4: Select Top Components
    
    Keep components that explain the most variance:
    - PC1 might explain 40% of variance
    - PC2 might explain 25%
    - PC3 might explain 15%
    - ... rest explain < 5% each (mostly noise)
    
    **Goal:** Keep enough PCs to capture 90-95% of total variance.
    """)
    
    st.markdown("---")
    st.subheader("🎯 What Are Principal Components?")
    
    st.markdown("""
    Principal components are **new features** created by combining original features.
    
    **Example:**
    
    **PC1** = 0.45×glucose + 0.40×insulin + 0.35×BMI + 0.30×age + ...
    
    **PC2** = 0.50×exercise - 0.45×stress + 0.35×BP + ...
    
    **Interpretation:**
    - **PC1** might represent "metabolic health" (glucose, insulin, weight)
    - **PC2** might represent "lifestyle factors" (exercise, stress)
    
    Each PC is a weighted combination of original features!
    """)
    
    st.markdown("---")
    st.subheader("✅ Strengths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Reduces overfitting** - Fewer features = simpler models
        - **Removes multicollinearity** - Correlated features combined
        - **Speeds up training** - Fewer dimensions = faster computation
        - **Noise reduction** - Removes low-variance components
        """)
    
    with col2:
        st.markdown("""
        - **Visualization** - Can plot data in 2D/3D
        - **Data compression** - Store less data
        - **Feature extraction** - Discover hidden patterns
        - **Preprocessing step** - Improves other models
        """)
    
    st.markdown("---")
    st.subheader("⚠️ Limitations")
    
    st.markdown("""
    - **Loss of interpretability** - PC1 isn't a real feature like "glucose"
    - **Linear transformations only** - Can't capture non-linear patterns
    - **Variance ≠ importance** - High variance doesn't always mean important
    - **Sensitive to scaling** - Must standardize features first
    - **Choosing components** - How much variance to keep?
    """)
    
    st.markdown("---")
    st.subheader("🏥 Application in Diabetes Type Classification")
    
    st.markdown("""
    PCA is used before SVM classification because:
    
    1. **12 features → 3-4 components**
       - Original: age, gender, glucose, insulin, BMI, BP, activity, stress, family history, fatigue, urination, thirst
       - PCA: PC1, PC2, PC3, PC4 (captures 90% of information)
    
    2. **Reveals hidden patterns**
       - PC1 often captures age-BMI relationship
       - Type 1: Young + low BMI → low PC1
       - Type 2: Older + high BMI → high PC1
    
    3. **Improves classification**
       - Removes noise and correlation
       - SVM works better in lower dimensions
       - Faster training and prediction
    
    **Real example from this app:**
    - Type 1 cluster: PC1 ≈ -0.83 (younger, leaner patients)
    - Type 2 cluster: PC1 ≈ +1.15 (older, heavier patients)
    
    Clear separation in just one component!
    """)

# SVM
elif model_section == "SVM":
    st.header("🎯 Support Vector Machine (SVM)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is SVM?
        
        Support Vector Machine is a **supervised learning algorithm** that finds the best boundary 
        (hyperplane) to separate different classes. It's like drawing the best line to separate 
        diabetic from non-diabetic patients.
        
        ### Key Concept: Maximum Margin
        
        SVM doesn't just find any separating line - it finds the line with the **maximum margin**:
        - Margin = distance between the line and the nearest data points
        - Larger margin = more confident separation = better generalization
        
        **Support Vectors** are the critical data points closest to the boundary that define the margin.
        """)
    
    with col2:
        st.info("""
        **Key Characteristics**
        
        ⚡ **Speed:** Medium-slow
        
        📊 **Data Needs:** Medium
        
        🎯 **Accuracy:** Very high
        
        🧠 **Interpretability:** Low-medium
        
        ⚙️ **Complexity:** Medium-high
        """)
    
    st.markdown("---")
    st.subheader("📊 How SVM Works")
    
    st.markdown("""
    ### 1. Linear SVM (Linearly Separable Data)
    
    ```
    Type 1 patients:  ●  ●  ●     |     Type 2 patients:  ○  ○  ○
                      ●  ●        |                       ○  ○
                                  |
                        Maximum Margin Boundary
    ```
    
    **Goal:** Find the line that maximizes the gap between classes.
    
    ### 2. Linear SVM (Non-Separable Data)
    
    Real data often has some overlap:
    
    ```
    Type 1:  ●  ●  ● ○ |     Type 2:  ○  ○  ○
             ●  ● ○    |              ○  ● ○
    ```
    
    **Solution:** Allow some misclassification (soft margin) with penalty C:
    - High C: Few mistakes allowed (risk overfitting)
    - Low C: More mistakes okay (simpler boundary)
    
    ### 3. Kernel Trick (Non-Linear Data)
    
    Sometimes no straight line can separate classes. SVM can use **kernels** to map data to higher dimensions:
    
    - **Linear kernel**: Standard straight-line separation
    - **RBF kernel**: Creates circular/curved boundaries
    - **Polynomial kernel**: Complex curved boundaries
    
    **In this app:** We use linear kernel because PCA already transformed the data.
    """)
    
    st.markdown("---")
    st.subheader("✅ Strengths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Effective in high dimensions** - Works well with many features
        - **Memory efficient** - Only stores support vectors
        - **Versatile** - Different kernels for different patterns
        - **Robust to overfitting** - Especially in high-dimensional space
        """)
    
    with col2:
        st.markdown("""
        - **Works with small datasets** - Doesn't need huge data
        - **Clear decision boundary** - Geometric interpretation
        - **Good generalization** - Maximum margin principle
        - **Handles outliers** - Soft margin allows mistakes
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Key Hyperparameters")
    
    st.markdown("""
    | Parameter | What It Does | Typical Values |
    |-----------|--------------|----------------|
    | **C (penalty)** | Trade-off between margin size and misclassification | 0.1 - 100 |
    | **kernel** | Type of decision boundary | 'linear', 'rbf', 'poly' |
    | **gamma** | Influence of single training point (for RBF/poly) | 'scale', 'auto', 0.001-1 |
    
    **In this app:** We use `kernel='linear'` since PCA already handles non-linearity.
    """)
    
    st.markdown("---")
    st.subheader("⚠️ Limitations")
    
    st.markdown("""
    - **Slow on large datasets** - Training time grows with data size
    - **Sensitive to feature scaling** - Must standardize features
    - **Choice of kernel** - Wrong kernel = poor performance
    - **No probability estimates** - Only gives class labels (unless calibrated)
    - **Black box with kernels** - Hard to interpret non-linear SVMs
    """)
    
    st.markdown("---")
    st.subheader("🏥 Application in Diabetes Type Classification")
    
    st.markdown("""
    SVM is used after PCA for type classification because:
    
    1. **Clear separation in PCA space**
       - Type 1 and Type 2 form distinct clusters
       - SVM finds the optimal boundary between them
    
    2. **Works with pseudo-labels**
       - K-Means creates initial Type 1/Type 2 labels
       - SVM learns from these to classify new patients
    
    3. **Robust classification**
       - Maximum margin ensures good generalization
       - Handles cases where types have some overlap
    
    4. **Efficient with PCA**
       - After PCA: only 3-4 dimensions
       - SVM trains fast in low dimensions
    
    **Process:**
    ```
    Patient features (12D) 
        → PCA (reduce to 3-4D)
        → SVM (find optimal boundary)
        → Prediction: Type 1 or Type 2
    ```
    """)

# Comparison
elif model_section == "Comparison":
    st.header("⚖️ Model Comparison & Selection Guide")
    
    st.markdown("""
    Choosing the right model depends on your specific needs. Here's a comprehensive comparison 
    to help you understand when to use each model.
    """)
    
    st.markdown("---")
    st.subheader("📊 Quick Comparison Table")
    
    comparison_data = {
        'Model': ['Naive Bayes', 'Random Forest', 'Logistic Regression', 'K-Means', 'PCA', 'SVM'],
        'Type': ['Supervised', 'Supervised', 'Supervised', 'Unsupervised', 'Unsupervised', 'Supervised'],
        'Speed': ['⚡⚡⚡', '⚡⚡', '⚡⚡⚡', '⚡⚡⚡', '⚡⚡', '⚡'],
        'Accuracy': ['Good', 'Very High', 'Very High', 'N/A', 'N/A', 'Very High'],
        'Interpretability': ['High', 'Medium', 'Very High', 'High', 'Low', 'Low-Medium'],
        'Data Needs': ['Small', 'Medium-Large', 'Small-Large', 'Medium', 'Medium-Large', 'Medium'],
        'Best For': ['Quick screening', 'Complex patterns', 'Clinical settings', 'Risk grouping', 'Feature reduction', 'Type classification']
    }
    
    st.table(comparison_data)
    
    st.markdown("---")
    st.subheader("🎯 When to Use Each Model")
    
    tab1, tab2, tab3 = st.tabs(["Classification", "Clustering", "Preprocessing"])
    
    with tab1:
        st.markdown("""
        ### Classification Models (Supervised)
        
        **Use Naive Bayes when:**
        - ✅ Need fast predictions (real-time screening)
        - ✅ Working with small datasets
        - ✅ Want probability outputs
        - ✅ Need a simple baseline
        - ❌ Avoid when features are highly dependent
        
        **Use Random Forest when:**
        - ✅ Have enough data (500+ samples)
        - ✅ Need highest accuracy
        - ✅ Want to understand feature importance
        - ✅ Data has non-linear patterns
        - ✅ Can tolerate longer training time
        - ❌ Avoid when interpretability is critical
        
        **Use Logistic Regression when:**
        - ✅ Need interpretable coefficients
        - ✅ Working in regulated/clinical environment
        - ✅ Want probability calibration
        - ✅ Have linear relationships
        - ✅ Need to explain to non-technical stakeholders
        - ❌ Avoid when relationships are highly non-linear
        
        **Use SVM when:**
        - ✅ Working in high-dimensional space
        - ✅ Have medium-sized dataset
        - ✅ Need maximum margin separation
        - ✅ Classes are well-separated
        - ❌ Avoid with very large datasets (slow)
        """)
    
    with tab2:
        st.markdown("""
        ### Clustering Models (Unsupervised)
        
        **Use K-Means when:**
        - ✅ No labeled data available
        - ✅ Want to discover natural groupings
        - ✅ Need to identify risk groups
        - ✅ Have spherical/compact clusters
        - ✅ Know approximate number of groups
        - ❌ Avoid with irregular cluster shapes
        - ❌ Avoid with very different cluster sizes
        
        **Use PCA when:**
        - ✅ Have too many features (10+)
        - ✅ Features are correlated
        - ✅ Want to reduce computational cost
        - ✅ Need to visualize high-dimensional data
        - ✅ Want to remove noise
        - ❌ Avoid when features are already uncorrelated
        - ❌ Avoid when interpretability of features is crucial
        """)
    
    with tab3:
        st.markdown("""
        ### Preprocessing Techniques
        
        **Feature Selection (Correlation Analysis):**
        - ✅ Use before any model
        - ✅ Remove irrelevant features
        - ✅ Improve model performance
        - ✅ Reduce overfitting
        
        **Feature Scaling (Standardization):**
        - ✅ Required for: Logistic Regression, SVM, PCA, K-Means
        - ✅ Not required for: Naive Bayes, Random Forest
        - ✅ Always safe to apply
        
        **Encoding (Label Encoding):**
        - ✅ Required for categorical features
        - ✅ Convert text to numbers
        - ✅ Apply before any ML model
        """)
    
    st.markdown("---")
    st.subheader("🏆 Model Performance Expectations")
    
    st.markdown("""
    ### Typical Accuracy Ranges for Diabetes Prediction
    
    | Model | Accuracy Range | When It Excels | When It Struggles |
    |-------|----------------|----------------|-------------------|
    | **Naive Bayes** | 75-85% | Small data, independent features | Correlated features |
    | **Random Forest** | 85-95% | Complex patterns, enough data | Small datasets, need interpretability |
    | **Logistic Regression** | 80-92% | Linear relationships, scaled features | Non-linear patterns |
    | **SVM** | 85-93% | High dimensions, clear separation | Large datasets, slow |
    
    **Note:** Actual performance depends on data quality, feature engineering, and hyperparameter tuning.
    """)
    
    st.markdown("---")
    st.subheader("🎓 Learning Curve")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🌱 Beginner-Friendly
        - Naive Bayes
        - K-Means
        - Logistic Regression
        
        Simple math, easy to understand
        """)
    
    with col2:
        st.markdown("""
        ### 🌿 Intermediate
        - Random Forest
        - PCA
        
        More complex, but intuitive concepts
        """)
    
    with col3:
        st.markdown("""
        ### 🌳 Advanced
        - SVM (with kernels)
        - Deep Learning
        
        Requires mathematical background
        """)
    
    st.markdown("---")
    st.subheader("💡 Best Practices")
    
    st.markdown("""
    1. **Start Simple**
       - Begin with Logistic Regression or Naive Bayes
       - Establish baseline performance
       - Understand your data
    
    2. **Try Multiple Models**
       - Don't rely on just one model
       - Compare performance metrics
       - Use ensemble methods (voting)
    
    3. **Feature Engineering First**
       - Good features > fancy models
       - Use correlation analysis
       - Remove noisy features
    
    4. **Tune Hyperparameters**
       - Use GridSearchCV or RandomizedSearchCV
       - Validate on separate test set
       - Avoid overfitting
    
    5. **Consider Domain Requirements**
       - Clinical: Interpretability matters (Logistic Regression)
       - Screening: Speed matters (Naive Bayes)
       - Research: Accuracy matters (Random Forest, SVM)
    
    6. **Validate Thoroughly**
       - Use cross-validation
       - Test on unseen data
       - Monitor both precision and recall
    """)

# Footer
st.markdown("---")
st.info("""
### 📖 Further Learning

Want to dive deeper? Check out these resources:

- **scikit-learn Documentation**: Comprehensive guides and examples
- **Pattern Recognition and Machine Learning** by Christopher Bishop
- **The Elements of Statistical Learning** by Hastie, Tibshirani, and Friedman
- **Machine Learning for Healthcare** - Coursera courses
- **Kaggle Diabetes Datasets** - Practice with real data

This app demonstrates practical applications of these algorithms in healthcare. 
Remember: Models are tools to assist medical professionals, not replace them!
""")
