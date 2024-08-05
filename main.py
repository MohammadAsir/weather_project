import argparse
import os

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights

from src.data import ImageFolderDataModule
from src.graph import plot_loss_curves
from src.models.Resnet_from_scratch import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from src.models.resnet_pre import ResnetPre
from src.models.vit import Vit
from src.module import ModelLightningModule

DATA_DIR = os.path.join(os.curdir, "data","raw")

parser = argparse.ArgumentParser(description="Training script with argparse")

parser.add_argument('--learning_rate', type=float, 
                    default=1.2e-4, help='Learning rate')
parser.add_argument('--max_epochs', type=int, 
                    default=10, help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int, 
                    default=32, help='Batch size')
parser.add_argument('--num_classes', type=int, 
                    default=11, help='Number of classes')
parser.add_argument('--model', type=str, default='resnet50', 
                    help='Model to use')

args = parser.parse_args()
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.max_epochs
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
MODEL = args.model
NUM_CLASSES = args.num_classes

if MODEL == "Vit":
    model = Vit(num_classes=NUM_CLASSES)
    transform = ViT_B_16_Weights.DEFAULT.transforms()
    
elif MODEL == "ResnetPre":
    model = ResnetPre()
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
    
elif MODEL == "ResNet18":
    model = ResNet18(num_classes=NUM_CLASSES, channels=3)
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
    
elif MODEL == "ResNet34":
    model = ResNet34(num_classes=NUM_CLASSES, channels=3)
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
    
elif MODEL == "ResNet50":
    model = ResNet50(num_classes=NUM_CLASSES, channels=3)
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
    
elif MODEL == "ResNet101":
    model = ResNet101(num_classes=NUM_CLASSES, channels=3)
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
    
elif MODEL == "ResNet152":
    model = ResNet152(num_classes=NUM_CLASSES, channels=3)
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
else:
    raise ValueError(f"Unknown model: {MODEL}")

pl_model = ModelLightningModule(model, learning_rate=LEARNING_RATE,
                                 num_classes=NUM_CLASSES)
data_module = ImageFolderDataModule(DATA_DIR, transform=transform)

wandb_logger = WandbLogger(name='wandb_logs', project='image_classification')
wandb_logger.watch(pl_model, log='all')

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001,
                                     patience=3, mode="min")
ckpt_callback = ModelCheckpoint(save_top_k=1, mode='max', monitor="val_f1")

trainer = pl.Trainer(callbacks=[early_stop_callback, ckpt_callback],
                      max_epochs=NUM_EPOCHS, devices=1, 
                      accelerator='gpu', logger=wandb_logger)

trainer.fit(pl_model, data_module)
trainer.test(pl_model, data_module, ckpt_path='best')

plot_loss_curves(wandb_logger)