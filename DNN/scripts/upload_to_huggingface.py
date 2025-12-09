"""
Upload trained models to Hugging Face Hub for free hosting.
This allows your Vercel app to access the models via API.

Prerequisites:
1. Create a Hugging Face account: https://huggingface.co/join
2. Get your API token: https://huggingface.co/settings/tokens
3. Install: pip install huggingface_hub transformers torch

Usage:
    python scripts/upload_to_huggingface.py
"""

import os
from pathlib import Path

def upload_distilbert_model():
    """Upload fine-tuned DistilBERT model to Hugging Face"""
    try:
        from huggingface_hub import HfApi, create_repo
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
        
        # Your Hugging Face username (change this!)
        USERNAME = input("Enter your Hugging Face username: ").strip()
        
        # Model paths to check (in order of preference)
        possible_paths = [
            Path(__file__).parent.parent / "outputs" / "transformer",
            Path("/Users/thinguyen/Library/CloudStorage/OneDrive-GrandCanyonUniversity/RECENT CLASSES/SHARED CLASSES/CST-435 JT/DNN_local/outputs/transformer"),
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                print(f"✅ Found model at: {path}")
                break
        
        if not model_path:
            print(f"❌ Model not found in any of these locations:")
            for path in possible_paths:
                print(f"   - {path}")
            print("\nTrain your model first or update the path in this script!")
            return
        
        # Repository name
        repo_name = f"{USERNAME}/sentiment-distilbert"
        
        print(f"\n📤 Uploading DistilBERT model to {repo_name}...")
        
        # Load model and tokenizer
        print("Loading model...")
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        
        # Create repository (if it doesn't exist)
        api = HfApi()
        try:
            create_repo(repo_name, exist_ok=True, private=False)
            print(f"✅ Repository created: https://huggingface.co/{repo_name}")
        except Exception as e:
            print(f"Note: {e}")
        
        # Push to hub
        print("Uploading model (this may take a few minutes)...")
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
        
        print(f"\n✅ DistilBERT model uploaded successfully!")
        print(f"🔗 Model URL: https://huggingface.co/{repo_name}")
        print(f"\n📝 Add this to your .env.local:")
        print(f"HUGGINGFACE_MODEL_ID={repo_name}")
        
        return repo_name
        
    except ImportError:
        print("❌ Required packages not installed.")
        print("Run: pip install huggingface_hub transformers torch")
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        print("\nMake sure you:")
        print("1. Created a Hugging Face account")
        print("2. Run: huggingface-cli login")
        print("3. Entered your access token")

def upload_rnn_model():
    """
    Upload RNN model to Hugging Face.
    Note: RNN models require custom code for loading.
    """
    print("\n📝 RNN Model Upload:")
    print("RNN models need custom loading code.")
    print("For simplicity, we'll use DistilBERT only in production.")
    print("The RNN model works perfectly in your local version!")

def main():
    print("=" * 60)
    print("🚀 Hugging Face Model Upload Tool")
    print("=" * 60)
    print("\nThis will upload your trained models to Hugging Face Hub")
    print("so your Vercel app can use them via API (free tier).\n")
    
    # Check if logged in
    print("First, make sure you're logged in to Hugging Face:")
    print("Run: huggingface-cli login")
    print("Then paste your access token from: https://huggingface.co/settings/tokens\n")
    
    response = input("Are you logged in? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\n1. Install CLI: pip install huggingface_hub")
        print("2. Login: huggingface-cli login")
        print("3. Run this script again")
        return
    
    # Upload DistilBERT
    model_id = upload_distilbert_model()
    
    # Info about RNN
    upload_rnn_model()
    
    if model_id:
        print("\n" + "=" * 60)
        print("✅ NEXT STEPS:")
        print("=" * 60)
        print("\n1. Create web/.env.local file with:")
        print(f"   HUGGINGFACE_MODEL_ID={model_id}")
        print("   HUGGINGFACE_TOKEN=your_token_here (optional for public models)")
        print("\n2. Add to Vercel environment variables:")
        print("   - Go to Vercel dashboard > Settings > Environment Variables")
        print(f"   - Add: HUGGINGFACE_MODEL_ID = {model_id}")
        print("\n3. Redeploy your Vercel app")
        print("\n4. Your app will now use the actual ML model! 🎉")

if __name__ == "__main__":
    main()
