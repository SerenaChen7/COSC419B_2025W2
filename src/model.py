"""
Jersey number classifier.

Supported architectures (--arch flag in train.py):
  mobilenet_v3_large  – default; good accuracy + CPU-friendly (5.5M params, ~3× faster than ResNet-18)
  mobilenet_v3_small  – fastest option, use if mobilenet_v3_large is still too slow
  resnet18            – original baseline (highest accuracy, 11.7M params)
"""
import torch
import torch.nn as nn
import torchvision.models as models

from dataset import NUM_CLASSES


def _is_head_param(name: str) -> bool:
    """Return True if a parameter belongs to the classification head."""
    return name.startswith('fc.') or name.startswith('classifier.')


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True,
                arch: str = 'mobilenet_v3_large') -> nn.Module:
    """
    Build a jersey-number classifier on top of a pretrained backbone.

    MobileNetV3-Small is the default: ~0.06 GFLOPs vs ~1.8 GFLOPs for
    ResNet-18 at 224×224, and roughly proportionally faster at 96×96.
    This makes it ~4-6× faster per forward pass on CPU while reaching
    comparable accuracy on digit-recognition tasks.
    """
    if arch == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # Replace only the final linear; keep the existing BN + Hardswish + Dropout neck
        in_features = model.classifier[3].in_features  # 1024
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif arch == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features  # 1280
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif arch == 'resnet18':
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

    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose: mobilenet_v3_small, mobilenet_v3_large, resnet18")

    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classification head."""
    for name, param in model.named_parameters():
        if not _is_head_param(name):
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def load_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    arch = state.get('args', {}).get('arch', 'resnet18')
    model = build_model(pretrained=False, arch=arch)
    model.load_state_dict(state['model_state_dict'])
    return model
