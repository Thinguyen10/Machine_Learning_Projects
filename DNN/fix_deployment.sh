#!/bin/bash

# Quick deployment fix script
# This automates the Vercel deployment with Hugging Face integration

set -e

echo "🚀 Vercel Deployment Fix Script"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the DNN project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python is not installed"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)

echo "Step 1: Checking for Hugging Face CLI..."
if ! $PYTHON_CMD -c "import huggingface_hub" 2>/dev/null; then
    echo "📦 Installing Hugging Face CLI..."
    pip install huggingface_hub transformers torch
else
    echo "✅ Hugging Face CLI already installed"
fi

echo ""
echo "Step 2: Checking Hugging Face login..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "❌ Not logged in to Hugging Face"
    echo ""
    echo "Please follow these steps:"
    echo "1. Sign up at: https://huggingface.co/join (if you don't have an account)"
    echo "2. Get your token at: https://huggingface.co/settings/tokens"
    echo "3. Run: huggingface-cli login"
    echo "4. Run this script again"
    exit 1
else
    echo "✅ Logged in to Hugging Face"
    HF_USER=$(huggingface-cli whoami | head -1 | awk '{print $1}')
    echo "   User: $HF_USER"
fi

echo ""
echo "Step 3: Checking if model exists..."
if [ ! -d "outputs/transformer" ]; then
    echo "❌ Model not found at outputs/transformer/"
    echo "   Please train your model first!"
    exit 1
else
    echo "✅ Model found"
fi

echo ""
echo "Step 4: Uploading model to Hugging Face..."
echo "This may take 2-3 minutes..."
$PYTHON_CMD scripts/upload_to_huggingface.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model uploaded successfully!"
    echo ""
    echo "📝 NEXT STEPS:"
    echo "=============="
    echo ""
    echo "1. Set environment variable in Vercel:"
    echo "   - Go to: https://vercel.com/dashboard"
    echo "   - Open your project"
    echo "   - Settings → Environment Variables"
    echo "   - Add: HUGGINGFACE_MODEL_ID = $HF_USER/sentiment-distilbert"
    echo ""
    echo "2. Deploy your code:"
    echo "   git add ."
    echo "   git commit -m 'Add Hugging Face integration'"
    echo "   git push origin main"
    echo ""
    echo "3. Wait for Vercel deployment to complete (2-3 min)"
    echo ""
    echo "4. Test your app!"
    echo "   - First request takes 20 seconds (model loading)"
    echo "   - Subsequent requests are fast (1-2 seconds)"
    echo ""
    echo "📚 For detailed instructions, see:"
    echo "   - QUICK_FIX.md (10-minute guide)"
    echo "   - STEP_BY_STEP.md (detailed walkthrough)"
    echo "   - DEPLOY_WITH_MODELS.md (complete documentation)"
    echo ""
else
    echo ""
    echo "❌ Model upload failed"
    echo "   See error messages above for details"
    exit 1
fi
