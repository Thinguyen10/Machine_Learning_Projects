# app_pretrained_pytorch.py - Streamlit app for PyTorch pre-trained model
# This app loads the PyTorch trained model and focuses on predictions

import streamlit as st
import kagglehub
import numpy as np
import pickle
import json
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from data_processing_pytorch import load_skin_cancer_data
from model_pytorch import SkinCancerCNN
from utils import plot_training_history
from streamlit_frontpage import show_front_page

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Detection - PyTorch CNN",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# FRONT PAGE
# ----------------------------
show_front_page()

# ----------------------------
# Sidebar Information
# ----------------------------
st.sidebar.header("ℹ️ Model Information")
st.sidebar.markdown("""
**Pre-Trained Model Details:**
- Framework: PyTorch
- Epochs Trained: 10
- Image Size: 64x64
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001
- Total Parameters: 1,143,113
""")

st.sidebar.markdown("---")
st.sidebar.header("🔧 Configuration")
batch_size = st.sidebar.selectbox("Batch Size for Data Loading", [16, 32, 64], index=1)
img_size = (64, 64)  # Fixed to match trained model

# ----------------------------
# STEP 1: Load Pre-Trained Model
# ----------------------------
st.header("Step 1: Load Pre-Trained PyTorch Model")
st.write("Load the CNN model that has been trained with PyTorch.")

MODEL_PATH = "trained_model_70epochs.pkl"
HISTORY_PATH = "training_history_70epochs.json"

if st.button("🔄 Load Pre-Trained Model"):
    try:
        # Load model data
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        # Recreate model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SkinCancerCNN(num_classes=model_data['num_classes'])
        model.load_state_dict(model_data['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        st.session_state['model'] = model
        st.session_state['device'] = device
        st.session_state['class_labels'] = model_data['class_labels']
        st.session_state['num_classes'] = model_data['num_classes']
        
        st.success("✅ Pre-trained PyTorch model loaded successfully!")
        
        # Load training history if available
        if Path(HISTORY_PATH).exists():
            with open(HISTORY_PATH, 'r') as f:
                history_dict = json.load(f)
            st.session_state['history_dict'] = history_dict
            
            # Display final metrics
            final_acc = history_dict['accuracy'][-1]
            final_val_acc = history_dict['val_accuracy'][-1]
            final_loss = history_dict['loss'][-1]
            final_val_loss = history_dict['val_loss'][-1]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Training Accuracy", f"{final_acc:.2%}")
            col2.metric("Final Validation Accuracy", f"{final_val_acc:.2%}")
            col3.metric("Final Training Loss", f"{final_loss:.4f}")
            col4.metric("Final Validation Loss", f"{final_val_loss:.4f}")
            
            st.info("📊 View training history in Step 5 below")
        
    except FileNotFoundError:
        st.error(f"❌ Model file '{MODEL_PATH}' not found! Please ensure the model has been trained and saved.")
        st.info("💡 Run `python train_quick_pytorch.py` to train the model first.")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

# ----------------------------
# STEP 2: Download Dataset
# ----------------------------
if 'model' in st.session_state:
    st.header("Step 2: Download Dataset")
    st.write("Download the Skin Cancer Detection dataset from Kaggle.")
    
    if st.button("📥 Download Dataset"):
        with st.spinner("Downloading dataset from KaggleHub..."):
            path = kagglehub.dataset_download("jaiahuja/skin-cancer-detection")
            DATA_DIR = Path(path)
            
            # Find dataset root
            def find_split_folder(base: Path):
                for child in base.rglob('*'):
                    if child.is_dir() and child.name.lower() in ['train', 'test']:
                        return child.parent
                return base
            
            DATA_ROOT = find_split_folder(DATA_DIR)
            st.session_state['data_root'] = DATA_ROOT
            st.success(f"✅ Dataset downloaded to: {DATA_ROOT}")

# ----------------------------
# STEP 3: Generate CSV Labels & Load Dataset
# ----------------------------
if 'data_root' in st.session_state:
    st.header("Step 3: Load Dataset")
    st.write("Generate CSV labels and load the dataset for predictions.")
    
    if st.button("📂 Load Dataset"):
        DATA_ROOT = Path(st.session_state['data_root'])
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
        
        def scan_folder(root_dir: Path):
            data = []
            for label_dir in sorted(root_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                label = label_dir.name
                for img_file in sorted(label_dir.rglob('*')):
                    if img_file.suffix.lower() in IMAGE_EXTS:
                        data.append((str(img_file.resolve()), label))
            return data
        
        # Generate CSVs
        csv_data = {}
        for split_dir in DATA_ROOT.iterdir():
            if split_dir.is_dir() and split_dir.name.lower() in ['train', 'test']:
                split_name = split_dir.name.lower()
                data = scan_folder(split_dir)
                if not data:
                    st.warning(f"No images found in {split_dir}")
                    continue
                df = pd.DataFrame(data, columns=['filepath', 'label'])
                csv_data[split_name] = df
        
        if csv_data:
            st.session_state['csv_data'] = csv_data
            
            # Load data loaders
            train_loader, val_loader, test_loader, class_labels = load_skin_cancer_data(
                df_train=csv_data['train'],
                df_test=csv_data['test'],
                img_size=img_size,
                batch_size=batch_size
            )
            
            st.session_state.train_loader = train_loader
            st.session_state.val_loader = val_loader
            st.session_state.test_loader = test_loader
            
            st.success(f"✅ Dataset loaded successfully!")
            st.write(f"**Classes detected ({len(class_labels)})**: {', '.join(class_labels)}")
            
            # Show dataset statistics
            col1, col2 = st.columns(2)
            col1.metric("Training Images", len(csv_data['train']))
            col2.metric("Test Images", len(csv_data['test']))
            
            # Preview sample images
            st.subheader("📸 Sample Training Images")
            
            # Get first batch
            images, labels = next(iter(train_loader))
            
            # Define inverse transform to display images
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            
            cols = st.columns(5)
            for i, col in enumerate(cols[:5]):
                if i < len(images):
                    # Denormalize image
                    img_tensor = inv_normalize(images[i])
                    img_array = img_tensor.permute(1, 2, 0).numpy()
                    img_array = np.clip(img_array, 0, 1)
                    
                    label_idx = labels[i].item()
                    label_name = class_labels[label_idx]
                    col.image(img_array, caption=label_name, use_container_width=True)

# ----------------------------
# STEP 4: Make Predictions
# ----------------------------
if 'test_loader' in st.session_state and 'model' in st.session_state:
    st.header("Step 4: Test Model Predictions")
    st.write("Click the button below to see predictions on random test images.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        num_predictions = st.slider("Number of predictions to show", 1, 10, 5)
    
    with col2:
        show_correct_only = st.checkbox("Show only correct predictions", value=False)
        show_incorrect_only = st.checkbox("Show only incorrect predictions", value=False)
    
    if st.button("🔮 Generate Predictions"):
        test_loader = st.session_state.test_loader
        model = st.session_state.model
        device = st.session_state.device
        class_labels = st.session_state['class_labels']
        
        # Define inverse transform
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        predictions_made = 0
        attempts = 0
        max_attempts = len(test_loader.dataset)
        
        st.subheader("🎯 Model Predictions")
        
        cols_per_row = 5
        prediction_cols = st.columns(cols_per_row)
        col_idx = 0
        
        # Get random samples from test set
        test_dataset = test_loader.dataset
        indices = np.random.choice(len(test_dataset), min(max_attempts, 50), replace=False)
        
        for idx in indices:
            if predictions_made >= num_predictions:
                break
                
            img_tensor, true_label_idx = test_dataset[idx]
            true_label_name = class_labels[true_label_idx]
            
            # Predict
            with torch.no_grad():
                img_batch = img_tensor.unsqueeze(0).to(device)
                outputs = model(img_batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                pred_probs = probabilities[0].cpu().numpy()
                pred_idx = pred_probs.argmax()
                pred_label = class_labels[pred_idx]
                pred_conf = pred_probs[pred_idx] * 100
            
            is_correct = (pred_idx == true_label_idx)
            
            # Filter based on checkboxes
            if show_correct_only and not is_correct:
                attempts += 1
                continue
            if show_incorrect_only and is_correct:
                attempts += 1
                continue
            
            # Denormalize image for display
            img_display = inv_normalize(img_tensor)
            img_array = img_display.permute(1, 2, 0).numpy()
            img_array = np.clip(img_array, 0, 1)
            
            # Display prediction
            with prediction_cols[col_idx]:
                st.image(img_array, use_container_width=True)
                
                if is_correct:
                    st.success(f"✅ **{pred_label}**")
                else:
                    st.error(f"❌ Predicted: **{pred_label}**")
                    st.write(f"True: **{true_label_name}**")
                
                st.write(f"Confidence: **{pred_conf:.1f}%**")
                
                # Show top 3 predictions
                top_3_idx = np.argsort(pred_probs)[-3:][::-1]
                with st.expander("Top 3 predictions"):
                    for idx in top_3_idx:
                        st.write(f"{class_labels[idx]}: {pred_probs[idx]*100:.1f}%")
            
            col_idx = (col_idx + 1) % cols_per_row
            predictions_made += 1
            attempts += 1
        
        st.markdown("---")

# ----------------------------
# STEP 5: Evaluate Model Performance
# ----------------------------
if 'test_loader' in st.session_state and 'model' in st.session_state:
    st.header("Step 5: Model Performance Evaluation")
    
    if st.button("📊 Evaluate on Test Set"):
        model = st.session_state.model
        device = st.session_state.device
        test_loader = st.session_state.test_loader
        
        model.eval()
        test_correct = 0
        test_total = 0
        
        with st.spinner("Evaluating model on test set..."):
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
        
        test_accuracy = test_correct / test_total
        test_loss = 1 - test_accuracy  # Approximate
        
        col1, col2 = st.columns(2)
        col1.metric("Test Accuracy", f"{test_accuracy:.2%}")
        col2.metric("Test Samples", test_total)
        
        st.session_state['test_metrics'] = {
            'accuracy': test_accuracy,
            'total': test_total,
            'correct': test_correct
        }

# ----------------------------
# STEP 6: View Training History
# ----------------------------
if 'history_dict' in st.session_state:
    st.header("Step 6: Training History Visualization")
    st.write("View the training progress over epochs.")
    
    if st.button("📈 Show Training Curves"):
        history_dict = st.session_state['history_dict']
        
        # Create summary statistics
        col1, col2, col3 = st.columns(3)
        
        best_val_acc = max(history_dict['val_accuracy'])
        best_val_acc_epoch = history_dict['val_accuracy'].index(best_val_acc) + 1
        
        col1.metric("Best Validation Accuracy", f"{best_val_acc:.2%}", f"Epoch {best_val_acc_epoch}")
        col2.metric("Final Training Accuracy", f"{history_dict['accuracy'][-1]:.2%}")
        col3.metric("Total Epochs", len(history_dict['accuracy']))
        
        # Plot training curves
        fig = plot_training_history(history_dict)
        st.pyplot(fig)
        
        # Show epoch-by-epoch data in expander
        with st.expander("📋 View Detailed Training Metrics"):
            epochs_data = {
                'Epoch': list(range(1, len(history_dict['accuracy']) + 1)),
                'Train Loss': [f"{x:.4f}" for x in history_dict['loss']],
                'Train Accuracy': [f"{x:.4f}" for x in history_dict['accuracy']],
                'Val Loss': [f"{x:.4f}" for x in history_dict['val_loss']],
                'Val Accuracy': [f"{x:.4f}" for x in history_dict['val_accuracy']]
            }
            df_history = pd.DataFrame(epochs_data)
            st.dataframe(df_history, use_container_width=True, height=400)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🩺 <strong>Skin Cancer Detection CNN (PyTorch)</strong> - Pre-Trained Model</p>
    <p><em>For educational and research purposes only. Not for medical diagnosis.</em></p>
</div>
""", unsafe_allow_html=True)
