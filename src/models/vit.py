import torch.nn as nn
from torchvision.models import ViT_B_16_Weights
from torchvision.models.vision_transformer import vit_b_16

class Vit(nn.Module):
    def __init__(self, num_classes):
        super(Vit, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)