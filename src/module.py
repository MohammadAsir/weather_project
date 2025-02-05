import lightning.pytorch as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchmetrics.classification import (
    Accuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)


class ModelLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, num_classes=11):
        super(ModelLightningModule, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = MulticlassF1Score(num_classes=num_classes,
                                     average='weighted')
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        f1 = self.f1(y_hat.softmax(dim=-1), y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        f1 = self.f1(y_hat.softmax(dim=-1), y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        f1 = self.f1(y_hat.softmax(dim=-1), y)

        preds = torch.argmax(y_hat, dim=1)
        self.conf_matrix.update(preds, y)

        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_f1', f1)
        return loss
    
    def on_test_end(self):
        conf_matrix = self.conf_matrix.compute().cpu().numpy()
        print("Confusion Matrix:\n", conf_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= 1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               'min')
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 5
            }
        }
