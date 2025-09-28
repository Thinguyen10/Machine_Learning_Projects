""" main_app.py ----------- Streamlit app that ties everything together. """ 
import streamlit as st 
import pandas as pd 
from streamlit_intro import show_app_intro
from data_loader import load_data 
from data_cleaner import clean_data 
from train_test_split import prepare_features_target, split_data 
from perceptron import create_perceptron, train_model 
from model_evaluation import evaluate_model 
from visualization import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_feature_importance
from hyperparameter_tuning import run_hyperparameter_search 
import matplotlib.pyplot as plt 

st.title("Perceptron Model App")
show_app_intro()

# === File Upload === 
uploaded_file = st.sidebar.file_uploader( "Upload CSV or Excel file", type=["csv", "xlsx", "xls"] ) 

if uploaded_file is not None: 
    try: 
        if uploaded_file.name.endswith(('.xlsx', '.xls')): 
            df = pd.read_excel(uploaded_file) 
        else: 
            df = pd.read_csv(uploaded_file) 
        st.success("✅ Custom dataset loaded successfully!") 
    except Exception as e: 
        st.error(f"❌ Error loading file: {e}") 
else: 
    st.info("No file uploaded. Using default breast cancer dataset.") 
    df = load_data() # Show first rows st.write("### Raw Data Preview", df.head())

# Show first rows 
st.write("### Raw Data Preview", df.head())
    
# === Clean Data ===
df = clean_data(df)

# === Feature + Target selection ===
st.sidebar.header("Feature/Target Selection")
all_columns = df.columns.tolist()

# Let user select target column
target_column = st.sidebar.selectbox("Select Target Column", all_columns, index=len(all_columns)-1)

# Make sure target is numeric
if df[target_column].dtype == object:
    try:
        df[target_column] = df[target_column].map({'No':0, 'Yes':1})
    except:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.exceptions import NotFittedError
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

# Features are all except target
X, y = prepare_features_target(df, target_column=target_column)

# Save feature names for visualization later
feature_names = df.drop(columns=[target_column]).columns.tolist()

# === Train/test split ===
X_train, X_test, y_train, y_test = split_data(X, y)

# === Hyperparameters ===
st.sidebar.header("Hyperparameters")
max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 1000, step=100)
eta0 = st.sidebar.selectbox("Learning Rate (eta0)", [0.1, 1.0, 10.0])

# === Train Button ===
if st.button("Train Model"):
    model = create_perceptron(max_iter=max_iter, eta0=eta0)
    model = train_model(model, X_train, y_train)

    # Evaluate
    acc, report, y_pred, metrics = evaluate_model(model, X_test, y_test)
    st.write(f"### Accuracy: {acc:.2f}")
    st.write("### Precision:", metrics["precision"])
    st.write("### Recall:", metrics["recall"])
    st.write("### F1 Score:", metrics["f1"])
    st.write("### Classification Report", report)

    # Visualization
    st.pyplot(plot_confusion_matrix(y_test, y_pred))
    st.pyplot(plot_roc_curve(model, X_test, y_test))
    st.pyplot(plot_precision_recall_curve(model, X_test, y_test))

    # Feature importance (if applicable)
    fig_importance = plot_feature_importance(model, feature_names)
    if fig_importance:
        st.pyplot(fig_importance)


# === Hyperparameter Search Button ===
if st.button("Run Hyperparameter Search"):
    best_params, best_score = run_hyperparameter_search(X_train, y_train)
    st.write("### Hyperparameter Search Results")
    st.write("Best Params:", best_params)
    st.write("Best Score:", best_score)

    # Retrain model with best hyperparameters
    model = create_perceptron(
        max_iter=best_params.get("max_iter", max_iter),
        eta0=best_params.get("eta0", eta0)
    )
    model = train_model(model, X_train, y_train)

    # Evaluate tuned model
    st.write("Results With Tuned Parameters")
    acc, report, y_pred, metrics = evaluate_model(model, X_test, y_test)
    st.write(f"### Accuracy: {acc:.2f}")
    st.write("### Precision:", metrics["precision"])
    st.write("### Recall:", metrics["recall"])
    st.write("### F1 Score:", metrics["f1"])
    st.write("### Classification Report", report)

    # Visualization
    st.pyplot(plot_confusion_matrix(y_test, y_pred))
    st.pyplot(plot_roc_curve(model, X_test, y_test))
    st.pyplot(plot_precision_recall_curve(model, X_test, y_test))

    # Feature importance (if applicable)
    fig_importance = plot_feature_importance(model, feature_names)
    if fig_importance:
        st.pyplot(fig_importance)
