#!/bin/bash

# 🚀 Safe Git Push Guide
# This script helps you push ONLY the code (not heavy model files)

echo "🔍 Checking repository status..."
echo "================================"
echo ""

cd "/Users/thinguyen/Library/CloudStorage/OneDrive-GrandCanyonUniversity/RECENT CLASSES/SHARED CLASSES/CST-435 JT/DNN"

# Check for large files that might be accidentally committed
echo "Scanning for large files (>10MB)..."
LARGE_FILES=$(git ls-files | xargs ls -l 2>/dev/null | awk '$5 > 10485760 {print $9, "(" $5/1048576 "MB)"}')

if [ ! -z "$LARGE_FILES" ]; then
    echo "❌ WARNING: Large files found in git:"
    echo "$LARGE_FILES"
    echo ""
    echo "These files should be removed from git tracking."
    echo "Run: git rm --cached <filename>"
    exit 1
fi

echo "✅ No large files detected in git staging"
echo ""

# Show what will be committed
echo "📝 Files to be committed:"
echo "========================"
git status --short
echo ""

# Calculate approximate size
echo "📊 Repository size estimate:"
du -sh .git 2>/dev/null | awk '{print "Git folder: " $1}'
echo ""

# Ask for confirmation
read -p "🚀 Ready to commit and push? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📦 Adding files to git..."
    git add .
    
    echo "💬 Committing changes..."
    git commit -m "Prepare for Vercel deployment - exclude heavy model files (280MB)"
    
    echo "🌐 Pushing to GitHub..."
    git push origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully pushed to GitHub!"
        echo ""
        echo "📋 Next steps:"
        echo "1. Check your repository: https://github.com/thinguyen-dev/CST-435"
        echo "2. Deploy to Vercel: https://vercel.com/new"
        echo "3. Set Root Directory: web"
        echo "4. Click Deploy"
        echo ""
        echo "📚 See VERCEL_DEPLOYMENT.md for detailed instructions"
    else
        echo ""
        echo "❌ Push failed. Please check the error message above."
    fi
else
    echo ""
    echo "⏸️  Push cancelled. No changes made."
fi
