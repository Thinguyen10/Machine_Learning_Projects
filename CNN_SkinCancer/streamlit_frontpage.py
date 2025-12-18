import streamlit as st

def show_front_page():
    st.markdown("<h1 style='text-align: center; color: white;'>🩺 Skin Cancer Detection with Pre-Trained CNN</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='font-size: 24px; color: white; line-height: 1.6;'>
    
    ## 🎯 About This Application
    
    This application uses a **pre-trained Convolutional Neural Network (CNN)** built with **PyTorch** 
    to classify dermoscopic skin lesion images into **9 different categories**. 
    
    The model is ready to use - just click the button below to start testing!
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='font-size: 24px; color: white; line-height: 1.6;'>
    
    ## 🧠 CNN Model Architecture
    
    Our model is a custom-built Convolutional Neural Network with the following structure:
    
    **Architecture Layers:**
    - **Conv Block 1**: 32 filters (3x3) + ReLU + MaxPool
    - **Conv Block 2**: 64 filters (3x3) + ReLU + MaxPool  
    - **Conv Block 3**: 128 filters (3x3) + ReLU + MaxPool
    - **Flatten Layer**: Converts 2D features to 1D
    - **Dense Layer 1**: 128 neurons + ReLU + Dropout (0.5)
    - **Output Layer**: 9 neurons (softmax for classification)
    
    **Model Specifications:**
    - **Total Parameters**: 1,143,113 (4.36 MB)
    - **Framework**: PyTorch
    - **Training Epochs**: 80
    - **Optimizer**: Adam (learning rate: 0.001)
    - **Loss Function**: Cross Entropy Loss
    - **Batch Size**: 32
    - **Image Size**: 64x64 pixels
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='font-size: 24px; color: white; line-height: 1.6;'>
    
    ## 📊 Dataset: Skin Cancer Detection
    
    **Source**: Kaggle (jaiahuja/skin-cancer-detection)
    
    **9 Skin Lesion Categories:**
    1. **Actinic Keratosis** - Precancerous scaly patches
    2. **Basal Cell Carcinoma** - Most common skin cancer
    3. **Dermatofibroma** - Benign fibrous nodule
    4. **Melanoma** - Most dangerous skin cancer
    5. **Nevus** - Common mole
    6. **Pigmented Benign Keratosis** - Non-cancerous growth
    7. **Seborrheic Keratosis** - Benign wart-like growth
    8. **Squamous Cell Carcinoma** - Second most common skin cancer
    9. **Vascular Lesion** - Blood vessel abnormality
    
    **Dataset Statistics:**
    - **Total Images**: ~2,357 dermoscopic images
    - **Training Images**: 2,239
    - **Test Images**: 118
    - **Image Type**: High-quality dermoscopic skin lesion photographs
    - **Format**: JPG/PNG files
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='font-size: 24px; color: white; line-height: 1.6;'>
    
    ## 🚀 How to Use This Application
    
    Click the **"Start Testing"** button below to:
    - See the model classify random test images
    - View prediction confidence and accuracy
    - Analyze model performance with a confusion matrix
    
    The model is already trained and ready to classify skin lesions instantly!
    
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Pro Tip**: The model works best on dermoscopic images similar to those in the training dataset.")
    
    st.markdown("---")
    
    st.warning("""
    ⚠️ **Medical Disclaimer**: This application is for educational and research purposes only. 
    It should NOT be used as a substitute for professional medical diagnosis. Always consult 
    with qualified healthcare professionals for proper skin lesion evaluation and treatment.
    """)
