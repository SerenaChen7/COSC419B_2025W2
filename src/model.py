"""
Jersey number classifier using a pre-trained ResNet-18 backbone.
"""
import torch
import torch.nn as nn
import torchvision.models as models

from dataset import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 with a two-layer classification head.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(256, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classification head."""
    for name, param in model.named_parameters():
        if not name.startswith('fc.'):
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def load_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return model
