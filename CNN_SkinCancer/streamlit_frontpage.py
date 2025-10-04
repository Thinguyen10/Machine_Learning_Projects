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
    st.markdown("---")