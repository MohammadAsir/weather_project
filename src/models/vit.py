import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
class Vit(nn.Module):
    def __init__(self):
        super(Vit, self).__init__()
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, 11)
        num_classes = 11
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)