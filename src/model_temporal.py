"""
Spatio-temporal jersey number recogniser.

Architecture (Balaji et al., arXiv:2309.06285, Section 3.2):
    ResNet-18 spatial encoder  ->  Bi-LSTM  ->  two digit heads (d1, d2)

Input:  (B, T, 3, H, W)  -- B sequences of T frames each
Output: (logits_d1, logits_d2)  both shape (B, 11)
        digit classes 0-9 + class 10 = blank (single-digit number / illegible)
"""
import torch
import torch.nn as nn
import torchvision.models as models

NUM_DIGIT_CLASSES = 11   # 0-9 + blank

SPATIAL_FEAT_DIM  = 512  # ResNet-18 penultimate layer output
LSTM_HIDDEN       = 128  # per direction; total output dim = 256
LSTM_LAYERS       = 1


class SpatialEncoder(nn.Module):
    """ResNet-18 backbone with the final FC removed (outputs 512-d features)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        # Keep everything up to (and including) the global avg-pool; drop FC
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W)  ->  (N, 512)
        return self.features(x).flatten(1)


class SpatioTemporalNetwork(nn.Module):
    """
    Full spatio-temporal model.

    Forward pass:
      1. Apply SpatialEncoder to every frame independently  -> (B, T, 512)
      2. Feed the sequence through a Bi-LSTM               -> (B, T, 256)
      3. Mean-pool over the time axis                       -> (B, 256)
      4. Two linear heads predict the tens and units digits separately
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = SpatialEncoder(pretrained=pretrained)
        self.bilstm = nn.LSTM(
            input_size=SPATIAL_FEAT_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        temporal_dim = LSTM_HIDDEN * 2   # 256
        self.head_d1 = nn.Linear(temporal_dim, NUM_DIGIT_CLASSES)
        self.head_d2 = nn.Linear(temporal_dim, NUM_DIGIT_CLASSES)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        feats = self.encoder(x.view(B * T, C, H, W))   # (B*T, 512)
        feats = feats.view(B, T, -1)                    # (B, T, 512)
        lstm_out, _ = self.bilstm(feats)                # (B, T, 256)
        temporal = lstm_out.mean(dim=1)                 # (B, 256)
        return self.head_d1(temporal), self.head_d2(temporal)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def jersey_to_digits(jersey_num: int) -> tuple[int, int]:
    """
    Map a jersey number to (d1, d2) digit class indices.

    Examples:
        5  -> (10, 5)   # blank tens, units = 5
        15 -> (1,  5)   # tens = 1, units = 5
        -1 -> (10, 10)  # both blank = illegible
    """
    if jersey_num == -1:
        return 10, 10
    if jersey_num < 10:
        return 10, jersey_num
    return jersey_num // 10, jersey_num % 10


def digits_to_jersey(d1: int, d2: int) -> int:
    """
    Decode (d1, d2) digit indices back to a jersey number.

    Examples:
        (10, 5)  -> 5
        (1,  5)  -> 15
        (10, 10) -> -1  (illegible)
    """
    if d1 == 10 and d2 == 10:
        return -1
    if d1 == 10:
        return d2
    return d1 * 10 + d2


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model: nn.Module, optimizer, epoch: int,
                    val_acc: float, args) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'args': vars(args),
        'model_type': 'temporal',
    }, path)


def load_checkpoint(path: str, device: torch.device) -> 'SpatioTemporalNetwork':
    state = torch.load(path, map_location=device)
    model = SpatioTemporalNetwork(pretrained=False)
    model.load_state_dict(state['model_state_dict'])
    return model
