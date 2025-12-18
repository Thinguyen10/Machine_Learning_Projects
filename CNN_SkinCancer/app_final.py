"""
Skin Cancer Detection App - Final Version with Navigation
"""
import streamlit as st
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import kagglehub
from sklearn.metrics import confusion_matrix, classification_report
import random

# Import local modules
from model_pytorch import SkinCancerCNN
from streamlit_frontpage import show_front_page

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Detection CNN",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Navy blue background */
    .stApp {
        background-color: #0a1929;
    }
    
    /* Center content with max width */
    .block-container {
        max-width: 1200px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Much larger base font size */
    .stMarkdown {
        font-size: 24px !important;
    }
    
    /* Headers - much bigger */
    h1 {
        font-size: 64px !important;
        color: white !important;
        font-weight: bold !important;
    }
    h2 {
        font-size: 48px !important;
        color: white !important;
        font-weight: bold !important;
    }
    h3 {
        font-size: 32px !important;
        color: white !important;
    }
    h4 {
        font-size: 26px !important;
        color: white !important;
    }
    
    /* All text white and bigger */
    p, li, span, div {
        font-size: 24px !important;
        color: white !important;
    }
    
    /* Buttons - huge and visible */
    .stButton>button {
        width: 100%;
        height: 80px;
        font-size: 32px;
        font-weight: bold;
        background-color: #fbbf24;
        color: #0a1929;
        border-radius: 12px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #f59e0b;
        transform: scale(1.02);
    }
    
    /* Metric cards */
    .metric-card {
        padding: 25px;
        border-radius: 10px;
        background-color: #ffffff;
        border: 3px solid #e0e0e0;
        text-align: center;
    }
    .metric-card h1, .metric-card h2, .metric-card h3, .metric-card h4, .metric-card p {
        color: #1f2937 !important;
    }
    
    /* Info/warning boxes */
    .stAlert {
        font-size: 22px !important;
        background-color: rgba(251, 191, 36, 0.1);
        border: 2px solid #fbbf24;
    }
    .stAlert p {
        font-size: 22px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-size: 24px !important;
        color: white !important;
        background-color: rgba(251, 191, 36, 0.2);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #fbbf24 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the pre-trained PyTorch model"""
    try:
        with open('trained_model_70epochs.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        num_classes = model_data['num_classes']
        class_labels = model_data['class_labels']
        img_size = model_data['img_size']
        
        # Create model and load weights
        model = SkinCancerCNN(num_classes=num_classes)
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        return model, class_labels, img_size, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None


@st.cache_data
def download_and_find_dataset():
    """Download dataset and find test images"""
    try:
        # Download dataset
        path = kagglehub.dataset_download("jaiahuja/skin-cancer-detection")
        data_dir = Path(path)
        
        # Find test folder
        test_folder = None
        for child in data_dir.rglob('*'):
            if child.is_dir() and child.name.lower() == 'test':
                test_folder = child
                break
        
        if not test_folder:
            return None, None
        
        # Collect all test images with their labels
        test_data = []
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
        
        for label_dir in sorted(test_folder.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for img_file in sorted(label_dir.rglob('*')):
                if img_file.suffix.lower() in IMAGE_EXTS:
                    test_data.append({
                        'filepath': str(img_file.resolve()),
                        'label': label
                    })
        
        return test_folder, test_data
    except Exception as e:
        st.error(f"❌ Error downloading dataset: {str(e)}")
        return None, None


def preprocess_image(image_path, img_size):
    """Preprocess image for model prediction"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image, image_tensor


def predict_image(model, image_tensor, class_labels, device):
    """Make prediction on a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_label = class_labels[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
        top3_predictions = [
            (class_labels[idx.item()], prob.item()) 
            for idx, prob in zip(top3_idx[0], top3_prob[0])
        ]
    
    return predicted_label, confidence_score, top3_predictions


def evaluate_model(model, test_data, class_labels, img_size, device):
    """Evaluate model on all test data and return confusion matrix data"""
    true_labels = []
    predicted_labels = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(test_data)
    for idx, item in enumerate(test_data):
        status_text.text(f"Evaluating: {idx + 1}/{total} images...")
        progress_bar.progress((idx + 1) / total)
        
        _, image_tensor = preprocess_image(item['filepath'], img_size)
        predicted_label, _, _ = predict_image(model, image_tensor, class_labels, device)
        
        true_labels.append(item['label'])
        predicted_labels.append(predicted_label)
    
    progress_bar.empty()
    status_text.empty()
    
    return true_labels, predicted_labels


def plot_confusion_matrix(true_labels, predicted_labels, class_labels):
    """Create and display confusion matrix"""
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Skin Cancer Classification', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig, cm


def show_random_test_page():
    """Page for testing individual random images"""
    st.markdown("<h1 style='text-align: center; color: white;'>🎲 Random Image Test</h1>", 
                unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model and dataset..."):
        model, class_labels, img_size, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'trained_model_70epochs.pkl' exists.")
        return
    
    # Download dataset
    with st.spinner("Loading test images..."):
        test_folder, test_data = download_and_find_dataset()
    
    if test_data is None or len(test_data) == 0:
        st.error("Failed to load test data.")
        return
    
    st.markdown("---")
    
    # Single Image Prediction Section
    st.markdown("<h2 style='font-size: 32px;'>🎲 Random Image Test</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎲 Generate Random Test", key="generate_test"):
            # Select random test image
            random_sample = random.choice(test_data)
            
            # Store in session state
            st.session_state.current_sample = random_sample
    
    # Display prediction if we have a current sample
    if 'current_sample' in st.session_state:
        sample = st.session_state.current_sample
        
        # Load and preprocess image
        original_image, image_tensor = preprocess_image(sample['filepath'], img_size)
        
        # Make prediction
        predicted_label, confidence, top3_predictions = predict_image(
            model, image_tensor, class_labels, device
        )
        
        true_label = sample['label']
        is_correct = (predicted_label == true_label)
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(original_image, use_container_width=True)
        
        with col2:
            # True label
            st.markdown(f"""
            <div class='metric-card' style='background-color: #e3f2fd;'>
                <h3>🎯 True Disease</h3>
                <h2>{true_label}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Predicted label
            color = "#d1fae5" if is_correct else "#fee2e2"
            text_color = "#065f46" if is_correct else "#991b1b"
            icon = "✅" if is_correct else "❌"
            
            st.markdown(f"""
            <div class='metric-card' style='background-color: {color}; color: {text_color};'>
                <h3 style='color: {text_color};'>{icon} Predicted Disease</h3>
                <h2 style='color: {text_color};'>{predicted_label}</h2>
                <h4 style='color: {text_color};'>Confidence: {confidence*100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Top 3 predictions
            with st.expander("🏆 View Top 3 Predictions"):
                for i, (label, prob) in enumerate(top3_predictions, 1):
                    st.write(f"{i}. **{label}**: {prob*100:.2f}%")
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🏠 Back to Home", key="back_home_test"):
            st.session_state.page = "home"
            st.rerun()
    with col3:
        if st.button("📊 Full Analysis →", key="go_analysis"):
            st.session_state.page = "analysis"
            st.rerun()


def show_analysis_page():
    """Page for full model analysis with confusion matrix"""
    st.markdown("<h1 style='text-align: center; color: white;'>📊 Full Model Analysis</h1>", 
                unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model and dataset..."):
        model, class_labels, img_size, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'trained_model_70epochs.pkl' exists.")
        return
    
    # Download dataset
    with st.spinner("Loading test images..."):
        test_folder, test_data = download_and_find_dataset()
    
    if test_data is None or len(test_data) == 0:
        st.error("Failed to load test data.")
        return
    
    st.markdown("---")
    
    st.markdown("<h2 style='color: white;'>Confusion Matrix & Performance Metrics</h2>", unsafe_allow_html=True)
    
    if st.button("🔍 Analyze All Test Images", key="conf_matrix"):
        with st.spinner("🔄 Evaluating model on all test images... This may take a moment..."):
            true_labels, predicted_labels = evaluate_model(
                model, test_data, class_labels, img_size, device
            )
        
        # Store in session state
        st.session_state.true_labels = true_labels
        st.session_state.predicted_labels = predicted_labels
        st.session_state.class_labels = class_labels
    
    # Display confusion matrix if available
    if 'true_labels' in st.session_state:
        true_labels = st.session_state.true_labels
        predicted_labels = st.session_state.predicted_labels
        class_labels_saved = st.session_state.class_labels
        
        # Calculate accuracy
        correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
        total = len(true_labels)
        accuracy = (correct / total) * 100
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Total Test Images</h3>
                <h1>{total}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='background-color: #d1fae5; color: #065f46;'>
                <h3 style='color: #065f46;'>Correct Predictions</h3>
                <h1 style='color: #065f46;'>{correct}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='background-color: #bbdefb;'>
                <h3>Overall Accuracy</h3>
                <h1>{accuracy:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Plot confusion matrix
        fig, cm = plot_confusion_matrix(true_labels, predicted_labels, class_labels_saved)
        st.pyplot(fig)
        
        # Classification report
        report = classification_report(
            true_labels, predicted_labels, 
            labels=class_labels_saved,
            output_dict=True,
            zero_division=0
        )
        
        # Convert to dataframe for better display
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(3)
        class_report = report_df.iloc[:-3]  # Exclude accuracy, macro avg, weighted avg
        
        with st.expander("📋 View Detailed Metrics by Class"):
            st.dataframe(
                class_report,
                use_container_width=True,
                height=400
            )
        
        # Key insights
        f1_scores = class_report['f1-score'].to_dict()
        best_class = max(f1_scores, key=f1_scores.get)
        worst_class = min(f1_scores, key=f1_scores.get)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='background-color: #d1fae5;'>
                <h4 style='color: #065f46;'>🏆 Best Class</h4>
                <h3 style='color: #065f46;'>{best_class}</h3>
                <p style='color: #065f46;'>F1: {f1_scores[best_class]:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='background-color: #fef3c7;'>
                <h4 style='color: #92400e;'>⚠️ Needs Work</h4>
                <h3 style='color: #92400e;'>{worst_class}</h3>
                <p style='color: #92400e;'>F1: {f1_scores[worst_class]:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🏠 Back to Home", key="back_home_analysis"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        if st.button("← 🎲 Random Test", key="back_test"):
            st.session_state.page = "testing"
            st.rerun()


def main():
    """Main app function with navigation"""
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Page routing
    if st.session_state.page == "home":
        show_front_page()
        
        # Navigation button at the bottom
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Testing", key="go_testing"):
                st.session_state.page = "testing"
                st.rerun()
    
    elif st.session_state.page == "testing":
        show_random_test_page()
    
    elif st.session_state.page == "analysis":
        show_analysis_page()


if __name__ == "__main__":
    main()
