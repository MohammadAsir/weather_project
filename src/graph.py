import matplotlib.pyplot as plt
import pandas as pd

import wandb


def plot_loss_curves(logger):
    api = wandb.Api()
    entity = logger.experiment.entity
    project = logger.experiment.project
    run_id = logger.experiment.id
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(samples=10000)
    metrics = pd.DataFrame(history)

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