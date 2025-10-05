"""
data_loader.py
--------------
Decorative UI section to introduce and advertise the Streamlit app.
"""

import streamlit as st

def show_app_intro():
    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1 style="color:#4CAF50;">🧠 Perceptron Model Explorer</h1>
            <p style="font-size:18px;">
                Welcome to the <b>Perceptron Model App</b> – an interactive tool 
                that helps you upload datasets, preprocess data, train perceptron models, 
                and evaluate performance with beautiful visualizations.  
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "🚀 This app is designed for students, researchers, and enthusiasts "
        "who want a hands-on way to experiment with perceptron models."
    )

    st.markdown(
        """
        ### 🔑 Features
        - 📂 Upload your own dataset (`CSV` or `Excel`)  
        - 🧹 Clean and preprocess data automatically  
        - ⚙️ Train a perceptron model with customizable hyperparameters  
        - 📊 Visualize results with confusion matrices and metrics  
        - 🔍 Run hyperparameter tuning to optimize your model  

        ---
        💡 *Tip: Start by uploading a dataset in the sidebar, or use the default Frog dataset if no file is provided.*
        """
    )
