#!/bin/bash

# NLP Sentiment Analyzer - Model Training Script
# This script trains both sklearn and keras models with optimized hyperparameters

echo "============================================================"
echo "🚀 Training NLP Sentiment Analysis Models"
echo "============================================================"
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.8+"
    exit 1
fi

# Check Python version
echo "✓ Checking Python environment..."
python --version

# Check for required packages
echo "✓ Checking dependencies..."
python -c "import sklearn, tensorflow, pandas, nltk" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies missing. Installing..."
    pip install -r backend/requirements.txt
fi

# Train models
echo ""
echo "============================================================"
echo "📊 Starting model training..."
echo "============================================================"
echo ""

python -m backend.train_models

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ SUCCESS! Models trained successfully"
    echo "============================================================"
    echo ""
    echo "Generated files:"
    ls -lh model_sklearn.joblib model_keras vect.joblib 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
    echo "Next steps:"
    echo "  1. Start backend:  uvicorn backend.main:app --reload --port 8000"
    echo "  2. Start frontend: cd frontend && npm run dev"
    echo "  3. Open browser:   http://localhost:5173"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "❌ Training failed!"
    echo "============================================================"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check that backend/sentiment_analysis.csv exists"
    echo "  2. Verify Python environment has all dependencies"
    echo "  3. Try: pip install scikit-learn tensorflow pandas nltk"
    echo ""
    exit 1
fi
