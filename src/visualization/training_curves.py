import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_comparison_curves(histories, model_names):
    """Plot comparison of training curves for multiple models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['train_loss', 'val_loss', 'train_mae', 'val_mae']
    titles = ['Training Loss', 'Validation Loss', 'Training MAE', 'Validation MAE']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        for history, name in zip(histories, model_names):
            if metric in history:
                ax.plot(history[metric], label=name)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
    
    plt.tight