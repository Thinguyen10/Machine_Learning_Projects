"""
Visualization utilities for the diabetes prediction app.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st


def plot_correlation_matrix(df, figsize=(12, 8)):
    """Plot correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    
    return fig


def plot_elbow_curve(k_range, sse, optimal_k=None, title='Elbow Method'):
    """Plot elbow curve for K-Means clustering."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_range, sse, marker='o', linewidth=2, markersize=8)
    
    if optimal_k:
        ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k = {optimal_k}')
        ax.legend()
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Sum of Squared Distances (SSE)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_range)
    
    return fig


def plot_pca_variance(cumulative_variance, n_components=None):
    """Plot cumulative explained variance for PCA."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linewidth=2)
    ax.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Threshold')
    
    if n_components:
        ax.axvline(x=n_components, color='green', linestyle='--', 
                  label=f'Selected Components = {n_components}')
    
    ax.set_xlabel('Number of Principal Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('PCA - Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_feature_importance(feature_names, importances):
    """Plot feature importances for tree-based models."""
    fig = go.Figure([go.Bar(
        x=importances,
        y=feature_names,
        orientation='h',
        marker=dict(
            color=importances,
            colorscale='Viridis',
            showscale=True
        )
    )])
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_metrics_comparison(metrics_dict):
    """Plot comparison of multiple models' metrics."""
    models = list(metrics_dict.keys())
    accuracy = [metrics_dict[m]['accuracy'] for m in models]
    precision = [metrics_dict[m]['precision'] for m in models]
    recall = [metrics_dict[m]['recall'] for m in models]
    f1 = [metrics_dict[m]['f1'] for m in models]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracy, marker_color='lightsalmon'))
    fig.add_trace(go.Bar(name='Precision', x=models, y=precision, marker_color='lightblue'))
    fig.add_trace(go.Bar(name='Recall', x=models, y=recall, marker_color='lightgreen'))
    fig.add_trace(go.Bar(name='F1 Score', x=models, y=f1, marker_color='plum'))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500,
        template='plotly_white',
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def plot_confusion_matrix_heatmap(cm, labels=['No Diabetes', 'Diabetes']):
    """Plot confusion matrix as heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_patient_profile_radar(patient_data, feature_names):
    """Create radar chart for patient profile visualization."""
    # Normalize data to 0-1 scale for better visualization
    patient_normalized = (patient_data - patient_data.min()) / (patient_data.max() - patient_data.min() + 1e-8)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=patient_normalized,
        theta=feature_names,
        fill='toself',
        name='Patient Profile',
        line=dict(color='rgb(0, 150, 255)'),
        fillcolor='rgba(0, 150, 255, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title='Patient Health Profile',
        height=500
    )
    
    return fig


def plot_distribution_comparison(df, feature, group_col='diabetes status'):
    """Plot distribution comparison between groups."""
    fig = go.Figure()
    
    for group in df[group_col].unique():
        data = df[df[group_col] == group][feature]
        label = 'Diabetic' if group == 1 else 'Non-Diabetic'
        
        fig.add_trace(go.Histogram(
            x=data,
            name=label,
            opacity=0.7,
            nbinsx=30
        ))
    
    fig.update_layout(
        title=f'{feature.title()} Distribution',
        xaxis_title=feature.title(),
        yaxis_title='Count',
        barmode='overlay',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_risk_distribution(risk_labels):
    """Plot pie chart for risk distribution."""
    risk_counts = pd.Series(risk_labels).value_counts()
    
    colors = ['#ff6b6b', '#51cf66']  # Red for high risk, green for low risk
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker=dict(colors=colors),
        hole=0.3
    )])
    
    fig.update_layout(
        title='Risk Level Distribution',
        height=400,
        template='plotly_white'
    )
    
    return fig


def display_metrics_cards(metrics):
    """Display metrics in styled cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.2%}")


def create_gauge_chart(value, title, max_value=1):
    """Create a gauge chart for single metric visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': 'lightgray'},
                {'range': [max_value * 0.5, max_value * 0.75], 'color': 'gray'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig
