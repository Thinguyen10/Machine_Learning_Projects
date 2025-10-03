import streamlit as st
import kagglehub
import os
import random
from PIL import Image

def show_front_page():
    st.title("🩺 Skin Cancer Detection Dataset Overview")

    st.markdown("""
    ## Overview
    The **Skin Cancer Detection** dataset (by Jai Ahuja on Kaggle) is designed for building 
    **Convolutional Neural Network (CNN)** models to classify skin lesions as **benign** or **malignant**.  
    The images are derived from the **International Skin Imaging Collaboration (ISIC)** archive.
    """)

    st.markdown("""
    ## Data Characteristics
    - **Number of images**: ~2,357
    - **Classes / Labels**: Benign, Malignant
    - **Image Source**: Dermoscopic / skin lesion images from ISIC archive
    - **Intended Use**: Training/testing CNNs or other ML models for lesion classification
    """)

    st.markdown("""
    ## Strengths
    - Open and accessible for reproducible research  
    - Clear binary classification task (benign vs malignant)  
    - Derived from a recognized source (ISIC archive)  
    """)

    st.markdown("""
    ## Limitations / Challenges
    - ⚠️ **Relatively small sample size** (2,357 images)  
    - ⚠️ **Possible class imbalance** (malignant vs benign not clearly specified)  
    - ⚠️ **Limited metadata** (mostly just image + label)  
    - ⚠️ **Generalization risk** if used in isolation  
    """)

    st.markdown("""
    ## Potential Uses
    - **Baseline model training** for CNNs  
    - **Data augmentation & preprocessing research** (rotations, flips, normalization)  
    - **Transfer learning** using pre-trained models (e.g., ImageNet)  
    - **Benchmarking & performance comparison** (accuracy, AUC, sensitivity, specificity)  
    - **Dataset expansion** by combining with other ISIC datasets  
    """)

    st.info("💡 Tip: Start with transfer learning and augmentation to get better results with limited data.")

    # ------------------------------
    # IMAGE PREVIEW SECTION
    # ------------------------------
    st.markdown("## 🔍 Sample Images from the Dataset")

    dataset_dir = "jaiahuja/skin-cancer-detection"  # adjust if your dataset path differs
    categories = ["benign", "malignant"]

    # Loop through each category and show random samples
    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        if not os.path.exists(category_path):
            st.warning(f"Directory not found: {category_path}")
            continue

        # Pick 2 random images from this category
        sample_images = random.sample(os.listdir(category_path), 2)

        st.subheader(f"Class: {category} ({len(os.listdir(category_path))} images)")
        cols = st.columns(2)  # 5 images in a row

        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(category_path, img_name)
            cols[i].image(img_path, caption=f"{category}", use_column_width=True)