# 🩺 Diabetes ML Prediction System

> **Personal Portfolio Project** - Educational demonstration of machine learning applications in healthcare

A professional Streamlit application demonstrating machine learning for diabetes prediction, risk assessment, and type classification.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`

**Usage:** Train models → Compare performance → Make predictions → Assess risk → Classify types

---

## 🌟 Features

### 🏥 Diabetes Prediction
- **3 ML Models**: Naive Bayes, Random Forest, Logistic Regression
- Side-by-side performance comparison • Real-time predictions • Interactive visualizations

### ⚠️ Risk Assessment  
- K-Means clustering for risk stratification • Automatic optimal cluster detection • Individual risk evaluation

### 🔬 Type Classification
- PCA dimensionality reduction + SVM • Type 1 vs Type 2 classification • Variance analysis

### 📚 Model Explanations
- Detailed algorithm explanations • When to use each model • ML best practices

## 🤖 Machine Learning

**Models:** Naive Bayes • Random Forest (GridSearchCV) • Logistic Regression • SVM • K-Means • PCA

**Techniques:** Feature selection • Hyperparameter tuning • Feature scaling

**Tech Stack:** Streamlit • scikit-learn • pandas • numpy • plotly • matplotlib • seaborn

---� Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup
```bash
# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`
# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Getting Started
1. **Home Page**: Overview of the application and features
2. **Navigate**: Use the sidebar to access different sections

### Diabetes Prediction
1. Click "Start Training" to generate data and train models
2. Compare performance metrics across models
3. Make predictions on custom patient inputs
4. Explore data distributions and correlations

### Risk Assessment
1. Generate non-diabetic patient data
2. Perform clustering to identify risk groups
3. Assess individual patient risk levels
4. Analyze risk factors and characteristics

### Type Classification
1. Train the PCA + SVM classifier
2. View explained variance analysis
3. Classify patients into Type 1 or Type 2
4. Learn about dimensionality reduction

### Model Explanations
- Read detailed explanations of each algorithm
- Understand when to use each model
- Learn ML concepts through practical examples

## 🧪 Machine Learning Models

### Supervised Learning
- **Naive Bayes**: Fast probabilistic classifier
- **Random Forest**: Ensemble of decision trees with feature importance
- **Logistic Regression**: Interpretable linear model with probability outputs
- **SVM**: Maximum margin classifier for type distinction

### Unsupervised Learning
- **K-Means**: Clustering for risk grouping
- **PCA**: Dimensionality reduction for feature extraction

### Techniques
- Feature selection via correlation analysis
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation
- Feature scaling and standardization

## 📊 Dataset

The application uses **synthesized diabetes data** based on medical research, including:

- **Patient Demographics**: Age, Gender
- **Health Metrics**: Glucose, BMI, Insulin, Blood Pressure
- **Lifestyle**: Physical Activity, Stress Level
- **Medical History**: Family History
- **Symptoms**: Fatigue, Frequent Urination, Excessive Thirst
- **Target**: Diabetes Status

**Size**: 1,000 patients (configurable)  
**Classes**: Balanced (50% diabetic, 50% non-diabetic)

## 🎨 Key Technologies

- **Streamlit**: Interactive web application framework
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive charts and graphs
- **kneed**: Elbow detection for clustering

## ⚡ Performance Optimizations

**Problem:** Cold starts on Streamlit Cloud caused 25+ minute load times.

**Solutions Implemented:**
- **Lazy Imports**: Heavy libraries (pandas, sklearn, plotly) load only when needed
- **Data Caching** (`@st.cache_data`): Data generation cached by parameters
- **Model Caching** (`@st.cache_resource`): Trained models persist across sessions
- **Minimal Startup**: Main page loads instantly, processing deferred to user actions

**Results:**
- Cold start: 25+ min → **2-5 min** (80-90% faster)
- Cached operations: **1-2 seconds** (99% faster)
- Page navigation: 5-10 sec → **<1 sec**

**Optimized Files:** All pages use lazy loading and caching for maximum performance.

---

## 🎓 Educational Value & Purpose

**This is a personal portfolio project** demonstrating software engineering and machine learning best practices.

**Demonstrates:** Modular design • ML model comparison • Data preprocessing • Interactive UI/UX • Performance optimization

**Perfect for:** Healthcare ML applications • Portfolio projects • Teaching data science • Production-ready code examples

## 📚 Algorithm References

**Algorithms implemented using scikit-learn:**
- **Naive Bayes**: Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 2011
- **Random Forest**: Breiman, L., "Random Forests", Machine Learning, 2001
- **Logistic Regression**: Standard implementation from scikit-learn
- **Support Vector Machines**: Vapnik, V., "The Nature of Statistical Learning Theory", 1995
- **K-Means Clustering**: Lloyd, S., "Least squares quantization in PCM", IEEE, 1982
- **PCA**: Pearson, K., "On Lines and Planes of Closest Fit to Systems of Points in Space", 1901

**Libraries:**
- scikit-learn (Pedregosa et al., 2011) - ML algorithms
- Streamlit - Web framework
- Plotly, Matplotlib, Seaborn - Visualizations

## 🤝 Contributing

This is a personal portfolio project, but suggestions are welcome! Areas for improvement: XGBoost/Neural Networks • SHAP explainability • Real datasets • Model persistence

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

