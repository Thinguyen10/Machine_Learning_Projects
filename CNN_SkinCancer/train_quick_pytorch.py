"""Quick test training with PyTorch - 10 epochs"""
import torch
import pickle
import json
import kagglehub
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from data_processing_pytorch import load_skin_cancer_data
from model_pytorch import create_model

def main():
    EPOCHS = 80
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    IMG_SIZE = (64, 64)
    MODEL_SAVE_PATH = "trained_model_70epochs.pkl"
    HISTORY_SAVE_PATH = "training_history_70epochs.json"
    
    print("="*60)
    print("TRAINING - 80 EPOCHS (PyTorch)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\n[1/6] Downloading dataset...")
    path = kagglehub.dataset_download("jaiahuja/skin-cancer-detection")
    DATA_DIR = Path(path)
    print(f"✅ Dataset: {DATA_DIR}")
    
    print("\n[2/6] Finding dataset structure...")
    def find_split_folder(base):
        for child in base.rglob('*'):
            if child.is_dir() and child.name.lower() in ['train', 'test']:
                return child.parent
        return base
    DATA_ROOT = find_split_folder(DATA_DIR)
    print(f"✅ Root: {DATA_ROOT}")
    
    print("\n[3/6] Generating CSV labels...")
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
    def scan_folder(root_dir):
        data = []
        for label_dir in sorted(root_dir.iterdir()):
            if not label_dir.is_dir(): continue
            label = label_dir.name
            for img_file in sorted(label_dir.rglob('*')):
                if img_file.suffix.lower() in IMAGE_EXTS:
                    data.append((str(img_file.resolve()), label))
        return data
    
    csv_data = {}
    for split_dir in DATA_ROOT.iterdir():
        if split_dir.is_dir() and split_dir.name.lower() in ['train', 'test']:
            split_name = split_dir.name.lower()
            data = scan_folder(split_dir)
            if data:
                df = pd.DataFrame(data, columns=['filepath', 'label'])
                csv_data[split_name] = df
                print(f"✅ {split_name.capitalize()}: {len(df)} entries")
    
    print("\n[4/6] Loading dataset...")
    train_loader, val_loader, test_loader, class_labels = load_skin_cancer_data(
        df_train=csv_data['train'], df_test=csv_data['test'],
        img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    num_classes = len(class_labels)
    
    print("\n[5/6] Creating model...")
    model, optimizer, criterion, device = create_model(num_classes, LEARNING_RATE)
    
    print(f"\n[6/6] Training {EPOCHS} epochs...")
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_accuracy = train_correct / train_total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= val_total
        val_accuracy = val_correct / val_total
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    print("\n"+"="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
    model_data = {'model_state_dict': model.state_dict(), 'num_classes': num_classes,
                  'class_labels': class_labels, 'img_size': IMG_SIZE}
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✅ Model saved: {MODEL_SAVE_PATH}")
    
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ History saved: {HISTORY_SAVE_PATH}")
    print("="*60)

if __name__ == '__main__':
    main()
