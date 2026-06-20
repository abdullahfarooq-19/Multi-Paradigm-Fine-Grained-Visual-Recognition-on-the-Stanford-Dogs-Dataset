"""
Module 3: Transfer Learning Model Definition
ResNet50 with custom classifier head for 120-breed dog classification
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=120, pretrained=True, freeze_early=True):
    """
    Create ResNet50 model with transfer learning configuration.
    
    Args:
        num_classes: Number of output classes (120 breeds)
        pretrained: Use ImageNet pretrained weights
        freeze_early: Freeze conv1-layer3, fine-tune layer4+fc
    
    Returns:
        Configured ResNet50 model
    """
    # Load pretrained ResNet50
    weights = 'IMAGENET1K_V2' if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Freeze early layers if specified
    if freeze_early:
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    
    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    model = create_model(num_classes=120)
    trainable, total = count_parameters(model)
    print(f"Model: ResNet50")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")
