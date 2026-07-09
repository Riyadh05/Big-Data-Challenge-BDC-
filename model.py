import torch.nn as nn
from torchvision import models

def get_model(model_name='efficientnet_b0', num_classes=3):
    if model_name.startswith('efficientnet'):
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    else:  # resnet50
        model = models.resnet50(weights='IMAGENET1K_V2')
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model