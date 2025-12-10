"""
Upload trained 7-class sentiment model to HuggingFace Hub

After training completes, run this to upload the model:
    python upload_7class_model.py

This makes the model publicly accessible and avoids Git LFS issues.
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration
MODEL_DIR = "../../outputs/transformer_7class"
REPO_NAME = "Thi144/sentiment-distilbert-7class"  # Change username if needed

def upload_model():
    """Upload model to HuggingFace Hub."""
    
    # Check if model exists
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model not found at {MODEL_DIR}")
        print("Please train the model first using train_multiclass.py")
        return
    
    print(f"Uploading model from {MODEL_DIR} to {REPO_NAME}...")
    
    try:
        # Create repository (will skip if already exists)
        api = HfApi()
        create_repo(REPO_NAME, exist_ok=True, private=False)
        print(f"✅ Repository created/verified: {REPO_NAME}")
        
        # Upload all files
        api.upload_folder(
            folder_path=MODEL_DIR,
            repo_id=REPO_NAME,
            commit_message="Upload 7-class sentiment model (-3 to +3 scale)",
            ignore_patterns=["*.pyc", "__pycache__", ".git"]
        )
        
        print(f"\n✅ Model uploaded successfully!")
        print(f"🔗 View at: https://huggingface.co/{REPO_NAME}")
        print(f"\n📝 To use in your app, update MODEL_ID to: '{REPO_NAME}'")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\nMake sure you're logged in:")
        print("  huggingface-cli login")

if __name__ == "__main__":
    # Check if logged in
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"Logged in as: {whoami['name']}")
        upload_model()
    except Exception as e:
        print("❌ Not logged in to HuggingFace")
        print("\nPlease run:")
        print("  pip install huggingface_hub")
        print("  huggingface-cli login")
