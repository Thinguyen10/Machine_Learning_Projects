"""
Simple script to upload DistilBERT model to Hugging Face
Run this after: huggingface-cli login
"""

from pathlib import Path
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from huggingface_hub import HfApi

# Configuration
USERNAME = "Thi144"  # Your Hugging Face username
MODEL_NAME = "sentiment-distilbert"
REPO_NAME = f"{USERNAME}/{MODEL_NAME}"

# Alternative: Use organization if available
# Uncomment this if you want to use the CST-435 org instead:
# REPO_NAME = "CST-435/sentiment-distilbert"

# Model path
MODEL_PATH = Path(__file__).parent.parent / "outputs" / "transformer"

print(f"📤 Uploading model from: {MODEL_PATH}")
print(f"📍 Destination: https://huggingface.co/{REPO_NAME}")
print()

# Check token first
api = HfApi()
try:
    user_info = api.whoami()
    print(f"✅ Logged in as: {user_info['name']}")
except Exception as e:
    print(f"❌ Not logged in properly. Run: huggingface-cli login")
    print(f"   Make sure to use a token with WRITE permissions")
    print(f"   Get token from: https://huggingface.co/settings/tokens")
    exit(1)

# Load model and tokenizer
print("Loading model...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

# Upload to Hugging Face Hub
print("Uploading to Hugging Face (this may take 2-3 minutes)...")
try:
    model.push_to_hub(REPO_NAME, private=False)
    tokenizer.push_to_hub(REPO_NAME, private=False)
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure your token has WRITE permissions")
    print("2. Create token at: https://huggingface.co/settings/tokens")
    print("3. Select 'Write' permission when creating the token")
    print("4. Run: huggingface-cli login (and paste the new token)")
    print("\nAlternatively, try using the organization:")
    print("   Change REPO_NAME to: 'CST-435/sentiment-distilbert'")
    exit(1)

# Upload to Hugging Face Hub
print("Uploading to Hugging Face (this may take 2-3 minutes)...")
model.push_to_hub(REPO_NAME)
tokenizer.push_to_hub(REPO_NAME)

print()
print("✅ Upload complete!")
print(f"🔗 Model URL: https://huggingface.co/{REPO_NAME}")
print()
print("📝 Next steps:")
print("1. Add to Vercel environment variables:")
print(f"   HUGGINGFACE_MODEL_ID={REPO_NAME}")
print()
print("2. Deploy your app:")
print("   git add .")
print("   git commit -m 'Add Hugging Face integration'")
print("   git push origin main")
