# 🩺 Diabetes ML Prediction System

A professional, modular Streamlit application that demonstrates machine learning applications for diabetes prediction, risk assessment, and type classification. Built with software engineering best practices and educational clarity.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

### 1. 🏥 Diabetes Prediction
- **Multiple ML Models**: Naive Bayes, Random Forest, Logistic Regression
- **Model Comparison**: Side-by-side performance metrics
- **Interactive Predictions**: Real-time diagnosis with custom patient inputs
- **Visual Analytics**: Correlation matrices, feature distributions, patient profiles

### 2. ⚠️ Risk Assessment
- **K-Means Clustering**: Groups non-diabetic patients by risk level
- **Elbow Method**: Automatic optimal cluster detection
- **Risk Stratification**: High-risk vs low-risk identification
- **Individual Assessment**: Personalized risk evaluation

### 3. 🔬 Type Classification
- **PCA + SVM**: Dimensionality reduction and classification
- **Type 1 vs Type 2**: Distinguishes diabetes types
- **Variance Analysis**: Shows explained variance by components
- **Educational Visualizations**: Understand PCA process

### 4. 📚 Model Explanations
- **Comprehensive Documentation**: Detailed explanations of all algorithms
- **How They Work**: Step-by-step breakdowns
- **When to Use**: Decision guides for model selection
- **Best Practices**: ML tips and recommendations

## 🏗️ Architecture

```
Diabetes_Prediction/
├── app.py                          # Main Streamlit application
├── src/                            # Core modules (high modularity)
│   ├── data_processing.py          # Data generation and preprocessing
│   ├── models.py                   # ML model classes
│   └── visualizations.py           # Plotting and visualization utilities
├── pages/                          # Streamlit multi-page components
│   ├── 1_🏥_Diabetes_Prediction.py
│   ├── 2_⚠️_Risk_Assessment.py
│   ├── 3_🔬_Type_Classification.py
│   └── 4_📚_Model_Explanations.py
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone or Download
```bash
cd path/to/Diabetes_Prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

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

## 🚢 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and branch
5. Set main file path: `app.py`
6. Click "Deploy"

### Deploy to Other Platforms

**Heroku**:
```bash
# Add Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Add setup.sh
mkdir -p .streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > .streamlit/config.toml
```

**Docker**:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🎓 Educational Value

This project demonstrates:

- **Software Engineering**: Modular design, separation of concerns, DRY principles
- **Machine Learning**: Multiple algorithms, model comparison, hyperparameter tuning
- **Data Science**: Feature engineering, data preprocessing, visualization
- **UI/UX**: Interactive interfaces, user-friendly navigation, visual feedback
- **Documentation**: Clear explanations, inline comments, comprehensive README

Perfect for:
- Learning ML applications in healthcare
- Understanding different algorithms
- Building portfolio projects
- Teaching data science concepts

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional ML models (XGBoost, Neural Networks)
- More visualization options
- SHAP values for explainability
- Real dataset integration
- Model persistence (save/load trained models)
- API endpoints for predictions

## ⚠️ Disclaimer

This application is for **educational purposes only**. It demonstrates machine learning techniques but should **never replace professional medical diagnosis**. Always consult qualified healthcare professionals for medical advice and treatment.

## 📝 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2026

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

## 📧 Contact

For questions, suggestions, or collaborations, feel free to reach out!

---

**Built with ❤️ using Streamlit, scikit-learn, and Python**

*Demonstrating the power of machine learning in healthcare*
