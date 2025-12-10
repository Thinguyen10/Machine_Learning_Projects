"""
Upload 7-class model files individually to HuggingFace Hub
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration
MODEL_DIR = "outputs/transformer_7class"
REPO_NAME = "Thi144/sentiment-distilbert-7class"

def upload_individual_files():
    """Upload each file individually to ensure they're all uploaded."""
    
    # Files to upload
    files_to_upload = [
        "config.json",
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "model_7class.pkl",
        "metrics_7class.json",
        "confusion_matrix_7class.png"
    ]
    
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"Logged in as: {whoami['name']}")
        
        # Create repository
        create_repo(REPO_NAME, exist_ok=True, private=False)
        print(f"✅ Repository: {REPO_NAME}\n")
        
        # Upload each file
        for filename in files_to_upload:
            filepath = os.path.join(MODEL_DIR, filename)
            if os.path.exists(filepath):
                print(f"📤 Uploading {filename}... ", end="")
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=filename,
                    repo_id=REPO_NAME,
                    commit_message=f"Add {filename}"
                )
                print("✅")
            else:
                print(f"⚠️  Skipping {filename} (not found)")
        
        print(f"\n✅ Upload complete!")
        print(f"🔗 View at: https://huggingface.co/{REPO_NAME}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    upload_individual_files()
