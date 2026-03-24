"""
Jersey number classifier using a pre-trained ResNet-18 backbone.
"""
import torch
import torch.nn as nn
import torchvision.models as models

from dataset import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 with a custom classification head.
    The final FC layer is replaced with a linear layer for `num_classes` outputs.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return model
