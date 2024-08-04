import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curves(logger):
    metrics_path = os.path.join(logger.save_dir, 
                                logger.name, 
                                f"version_{logger.version}",
                                "metrics.csv")
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at: {metrics_path}")

    metrics = pd.read_csv(metrics_path)

    plt.figure(figsize=(10, 5))
    train_loss = metrics.dropna(subset=['train_loss'])
    val_loss = metrics.dropna(subset=['val_loss'])
    
    plt.plot(train_loss['epoch'], train_loss['train_loss'], 
             label='Training Loss')
    plt.plot(val_loss['epoch'], val_loss['val_loss'], 
             label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()