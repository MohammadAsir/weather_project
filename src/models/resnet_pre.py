import torch
import torch.nn as nn

class ResnetPre(nn.Module):
    def __init__(self):
        super(ResnetPre, self).__init__()
        self.model = torch.hub.load("pytorch/vision", "resnet50", 
                                    weights="IMAGENET1K_V2", 
                                    force_reload=False)
        num_classes = 11
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)