"""
Jersey number classifier.

Supported architectures (--arch):
    resnet34             default; best accuracy for jersey digit recognition
    resnet18             lighter alternative
    mobilenet_v3_small   fastest on CPU
    mobilenet_v3_large   better accuracy than small, moderately slower
"""
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 100  # class 0 = illegible, class 1-99 = jersey number


def build_model(num_classes: int = NUM_CLASSES,
                pretrained: bool = True,
                arch: str = 'resnet34') -> nn.Module:
    """Return a pretrained backbone with a replaced classification head."""

    if arch == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif arch == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
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

    elif arch == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features  # 512
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    else:
        raise ValueError(
            f"Unknown arch '{arch}'. Choose: resnet34, resnet18, mobilenet_v3_small, mobilenet_v3_large"
        )

    return model


def _is_head_param(name: str) -> bool:
    return name.startswith('fc.') or name.startswith('classifier.')




def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if not _is_head_param(name):
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def load_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    arch = state.get('args', {}).get('arch', 'resnet18')
    model = build_model(pretrained=False, arch=arch)
    model.load_state_dict(state['model_state_dict'])
    return model
