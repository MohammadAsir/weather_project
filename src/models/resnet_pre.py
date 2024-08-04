import torch
import torch.nn as nn
from torchvision import transforms
class ResnetPre(nn.Module):
    def __init__(self):
        super(ResnetPre, self).__init__()
        self.model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2", force_reload=False)
        num_classes = 11
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def transform(self, x):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(x)