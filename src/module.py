import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, MulticlassF1Score, MulticlassConfusionMatrix
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

LEARNING_RATE = 1.2e-4

class ResNetLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=LEARNING_RATE, num_classes=11):
        super(ResNetLightningModule, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average='weighted')
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

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

    
    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.05, patience=1, mode="min")
        ckpt_callback = ModelCheckpoint(save_top_k=1, mode='max', monitor="val_f1") 
        return [early_stop_callback, ckpt_callback]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

print("Module loaded")