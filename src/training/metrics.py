import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """Calculate various regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    
    # MAE
    axes[0, 1].plot(history['train_mae'], label='Train')
    axes[0, 1].plot(history['val_mae'], label='Validation')
    axes[0, 1].set_title('MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].legend()
    
    # RMSE
    axes[1, 0].plot(history['train_rmse'], label='Train')
    axes[1, 0].plot(history['val_rmse'], label='Validation')
    axes[1, 0].set_title('RMSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Root Mean Squared Error')
    axes[1, 0].legend()
    
    # Learning rate
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.show()