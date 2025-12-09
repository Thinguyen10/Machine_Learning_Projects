#!/bin/bash

echo "🚀 Deploying DNN Sentiment Analysis to Vercel"
echo "=============================================="
echo ""

# Check if in correct directory
if [ ! -d "web" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   Current: $(pwd)"
    echo "   Expected: .../CST-435 JT/DNN"
    exit 1
fi

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Navigate to web directory
cd web

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build locally to test
echo "🔨 Building project locally..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please fix errors before deploying."
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
cd ..
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your site should be live at the URL shown above"
