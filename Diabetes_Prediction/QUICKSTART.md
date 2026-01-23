# ⚡ Quick Start Guide

Get your Diabetes ML Prediction System running in minutes!

## 🚀 Fastest Way to Run

### Option 1: One Command (macOS/Linux)

```bash
cd "/Users/thinguyen/Documents/CLASSES/OLD CLASSES/CST-425/Projects/Diabetes_Prediction" && \
python -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
streamlit run app.py
```

### Option 2: Step by Step

```bash
# 1. Navigate to project
cd "Diabetes_Prediction"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## 📱 Using the App

### Home Page
- Read the overview
- Understand what the app does

### 🏥 Diabetes Prediction
1. Click **"Start Training"** button
2. Wait for models to train (~10-30 seconds)
3. Compare model performance metrics
4. Go to **"Make Predictions"** tab
5. Adjust sliders for patient information
6. Click **"Predict"** to see results

### ⚠️ Risk Assessment
1. Click **"Start Risk Assessment"** button
2. View elbow curve and optimal clusters
3. Go to **"Individual Assessment"** tab
4. Enter patient details
5. Click **"Assess Risk"** to see classification

### 🔬 Type Classification
1. Click **"Start Classification"** button
2. View PCA variance analysis
3. Go to **"Classify Patient"** tab
4. Enter complete patient profile
5. Click **"Classify Diabetes Type"**

### 📚 Model Explanations
- Select a model from the sidebar
- Read detailed explanations
- Understand when to use each algorithm

## 🎨 Customization

### Change Dataset Size
- Look for sliders in the sidebar
- Adjust between 500-2000 patients
- Larger = more training time but more robust

### Enable Hyperparameter Tuning
- Check "Tune Random Forest Hyperparameters"
- Warning: Significantly slower but better accuracy

### Adjust Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#your-color"  # Change to your preference
```

## 🐛 Troubleshooting

### App Won't Start
```bash
# Check Python version (need 3.9+)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Import Errors
```bash
# Make sure you're in the project directory
pwd

# Check if src/ directory exists
ls -la
```

### Page Not Found
- Make sure you're in the correct directory
- Check that `pages/` folder exists with all 4 page files

### Model Training Fails
- Reduce dataset size to 500
- Check console for specific error messages
- Ensure all dependencies are installed

## 💡 Pro Tips

1. **Start with small data**: Use 500-1000 samples for faster testing
2. **Train once, test many**: After training, make multiple predictions without retraining
3. **Explore all tabs**: Each page has multiple tabs with different features
4. **Read explanations**: Check the Model Explanations page to understand algorithms
5. **Compare models**: Use the Model Comparison tab to see which performs best

## 📊 Expected Results

### Diabetes Prediction
- Naive Bayes: ~75-85% accuracy
- Random Forest: ~85-95% accuracy
- Logistic Regression: ~80-92% accuracy

### Risk Assessment
- Usually finds 2-4 risk clusters
- 10-20% typically classified as high risk

### Type Classification
- PCA reduces to 3-5 components
- 90%+ variance retained
- Clear separation between types

## ⏱️ Performance

| Operation | Time (typical) |
|-----------|---------------|
| Data Generation (1000 samples) | 1-2 seconds |
| Model Training (3 models) | 5-15 seconds |
| Risk Clustering | 3-5 seconds |
| Type Classification | 5-10 seconds |
| Single Prediction | <1 second |

## 🎯 Next Steps

1. **Experiment**: Try different patient profiles
2. **Compare**: See how different models handle edge cases
3. **Learn**: Read model explanations thoroughly
4. **Customize**: Modify parameters and observe effects
5. **Deploy**: Follow DEPLOYMENT.md to share your app
6. **Extend**: Add new features or models

## 📚 Resources

- **README.md**: Complete documentation
- **DEPLOYMENT.md**: Deployment instructions
- **Model Explanations Page**: In-app learning materials
- **Streamlit Docs**: https://docs.streamlit.io

## ✅ Checklist for First Run

- [ ] Python 3.9+ installed
- [ ] In correct directory
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] App running (`streamlit run app.py`)
- [ ] Browser opened to http://localhost:8501
- [ ] Home page loads successfully
- [ ] Can navigate between pages
- [ ] Can train models in Diabetes Prediction page
- [ ] All visualizations render correctly

## 🎉 You're Ready!

Start exploring the app and learning about machine learning applications in healthcare!

**Questions?** Check the full README.md or Model Explanations page in the app.
