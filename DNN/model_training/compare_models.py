# Compare all three models and create comparison plots
# Compares RNN, Transformer (DistilBERT), and TF-IDF Baseline

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# Load metrics
def load_metrics():
    """Load metrics from all three models."""
    metrics = {}
    
    # RNN metrics
    try:
        with open('../outputs/rnn_metrics.json', 'r') as f:
            metrics['RNN (LSTM + Attention)'] = json.load(f)
    except FileNotFoundError:
        print("Warning: RNN metrics not found")
        metrics['RNN (LSTM + Attention)'] = None
    
    # Transformer metrics
    try:
        with open('../outputs/distilbert_metrics.json', 'r') as f:
            metrics['DistilBERT (Transformer)'] = json.load(f)
    except FileNotFoundError:
        print("Warning: DistilBERT metrics not found")
        metrics['DistilBERT (Transformer)'] = None
    
    # Baseline metrics
    try:
        with open('../outputs/baseline_metrics.json', 'r') as f:
            metrics['TF-IDF + Logistic Regression'] = json.load(f)
    except FileNotFoundError:
        print("Warning: Baseline metrics not found")
        metrics['TF-IDF + Logistic Regression'] = None
    
    return metrics

def plot_metrics_comparison(metrics):
    """Create comparison bar chart for all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison: RNN vs Transformer vs Baseline', fontsize=16, fontweight='bold')
    
    # Extract metrics
    models = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model_name, model_metrics in metrics.items():
        if model_metrics:
            models.append(model_name)
            accuracies.append(model_metrics['test_accuracy'])
            precisions.append(model_metrics['precision'])
            recalls.append(model_metrics['recall'])
            f1_scores.append(model_metrics['f1_score'])
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Accuracy
    axes[0, 0].bar(models, accuracies, color=colors[:len(models)])
    axes[0, 0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim([0.7, 1.0])
    axes[0, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Precision
    axes[0, 1].bar(models, precisions, color=colors[:len(models)])
    axes[0, 1].set_title('Precision', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim([0.7, 1.0])
    axes[0, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(precisions):
        axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Recall
    axes[1, 0].bar(models, recalls, color=colors[:len(models)])
    axes[1, 0].set_title('Recall', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim([0.7, 1.0])
    axes[1, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(recalls):
        axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    # F1 Score
    axes[1, 1].bar(models, f1_scores, color=colors[:len(models)])
    axes[1, 1].set_title('F1 Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0.7, 1.0])
    axes[1, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(f1_scores):
        axes[1, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../outputs/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Model comparison saved to outputs/figures/model_comparison.png")
    plt.close()

def plot_confusion_matrices(metrics):
    """Plot confusion matrices for all models side by side."""
    valid_models = {k: v for k, v in metrics.items() if v is not None}
    n_models = len(valid_models)
    
    if n_models == 0:
        print("No models to compare")
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
    
    for idx, (model_name, model_metrics) in enumerate(valid_models.items()):
        cm = np.array(model_metrics['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    ax=axes[idx], cbar=True)
        
        axes[idx].set_title(f'{model_name}\nAccuracy: {model_metrics["test_accuracy"]:.4f}',
                           fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('../outputs/figures/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print("Confusion matrices comparison saved to outputs/figures/confusion_matrices_comparison.png")
    plt.close()

def plot_training_curves(metrics):
    """Plot training curves for RNN (if available)."""
    rnn_metrics = metrics.get('RNN (LSTM + Attention)')
    
    if not rnn_metrics or 'train_losses' not in rnn_metrics:
        print("RNN training history not available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RNN Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(rnn_metrics['train_losses']) + 1)
    
    # Loss
    ax1.plot(epochs, rnn_metrics['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, rnn_metrics['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, rnn_metrics['train_accs'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, rnn_metrics['val_accs'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../outputs/figures/rnn_training_curves.png', dpi=300, bbox_inches='tight')
    print("RNN training curves saved to outputs/figures/rnn_training_curves.png")
    plt.close()

def create_summary_table(metrics):
    """Create and save summary table."""
    summary = "MODEL COMPARISON SUMMARY\n"
    summary += "=" * 80 + "\n\n"
    
    for model_name, model_metrics in metrics.items():
        if model_metrics:
            summary += f"{model_name}\n"
            summary += "-" * 80 + "\n"
            summary += f"  Test Accuracy:  {model_metrics['test_accuracy']:.4f}\n"
            summary += f"  Precision:      {model_metrics['precision']:.4f}\n"
            summary += f"  Recall:         {model_metrics['recall']:.4f}\n"
            summary += f"  F1 Score:       {model_metrics['f1_score']:.4f}\n"
            
            if 'hyperparameters' in model_metrics:
                summary += f"\n  Hyperparameters:\n"
                for param, value in model_metrics['hyperparameters'].items():
                    summary += f"    {param}: {value}\n"
            
            summary += "\n"
    
    # Save to file
    with open('../outputs/model_comparison_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n" + summary)
    print("Summary saved to outputs/model_comparison_summary.txt")

def main():
    """Main comparison pipeline."""
    print("=" * 80)
    print("MODEL COMPARISON: RNN vs Transformer vs Baseline")
    print("=" * 80)
    
    # Create output directory
    os.makedirs('../outputs/figures', exist_ok=True)
    
    # Load metrics
    print("\nLoading metrics...")
    metrics = load_metrics()
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    plot_metrics_comparison(metrics)
    plot_confusion_matrices(metrics)
    plot_training_curves(metrics)
    
    # Create summary
    print("\nCreating summary...")
    create_summary_table(metrics)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - outputs/figures/model_comparison.png")
    print("  - outputs/figures/confusion_matrices_comparison.png")
    print("  - outputs/figures/rnn_training_curves.png")
    print("  - outputs/model_comparison_summary.txt")

if __name__ == "__main__":
    main()
