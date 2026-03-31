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
LSTM_LAYERS       = 2    # inter-layer LSTM dropout activates when > 1


class AttentionPool(nn.Module):
    """
    Single-head additive attention over the time axis.
    Learns a context vector: score_t = w^T tanh(h_t), weight_t = softmax(scores).
    Output is the weighted sum of LSTM hidden states — emphasises informative frames.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (B, T, hidden_dim)
        scores  = self.attn(torch.tanh(lstm_out))      # (B, T, 1)
        weights = torch.softmax(scores, dim=1)          # (B, T, 1)
        return (weights * lstm_out).sum(dim=1)          # (B, hidden_dim)


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

    def __init__(self, pretrained: bool = True, dropout: float = 0.0,
                 lstm_layers: int = LSTM_LAYERS):
        super().__init__()
        self.encoder = SpatialEncoder(pretrained=pretrained)
        self.feat_dropout = nn.Dropout(p=dropout)   # applied to encoder output before LSTM
        self.bilstm = nn.LSTM(
            input_size=SPATIAL_FEAT_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn_pool = AttentionPool(LSTM_HIDDEN * 2)
        self.dropout = nn.Dropout(p=dropout)
        temporal_dim = LSTM_HIDDEN * 2   # 256
        self.head_d1 = nn.Linear(temporal_dim, NUM_DIGIT_CLASSES)
        self.head_d2 = nn.Linear(temporal_dim, NUM_DIGIT_CLASSES)

    def enable_mc_dropout(self, rate: float | None = None,
                          include_feat_dropout: bool = False) -> None:
        """
        Prepare the model for MC Dropout test-time augmentation.

        Sets the model to eval mode (so BatchNorm uses its running statistics,
        not batch statistics) while keeping selected Dropout layers in training
        mode so they sample stochastically across TTA passes.

        By default only the final dropout (before the digit heads) is made
        stochastic.  Enabling feat_dropout too is NOT recommended: it
        corrupts LSTM inputs across all frames, compounding noise over the
        sequence and drastically hurting accuracy.

        Parameters
        ----------
        rate : float, optional
            If given, overrides the Dropout probability for the enabled
            layers.  Use a value lower than the training rate (e.g. 0.1)
            to reduce per-pass noise when averaging few passes.
        include_feat_dropout : bool
            If True, also enables the feature dropout applied before the
            LSTM.  Default False — keeps LSTM inputs clean.
        """
        self.eval()
        targets = [self.dropout]  # final dropout before heads (always)
        if include_feat_dropout:
            targets.append(self.feat_dropout)
        for m in targets:
            m.train()
            if rate is not None:
                m.p = rate

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        feats = self.encoder(x.view(B * T, C, H, W))   # (B*T, 512)
        feats = feats.view(B, T, -1)                    # (B, T, 512)
        feats = self.feat_dropout(feats)                # dropout before LSTM
        lstm_out, _ = self.bilstm(feats)                # (B, T, 256)
        temporal = self.attn_pool(lstm_out)             # (B, 256) — attention over frames
        temporal = self.dropout(temporal)               # dropout before heads
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
    state = torch.load(path, map_location=device, weights_only=False)
    saved_args = state.get('args', {})
    dropout     = saved_args.get('dropout',     0.0)
    lstm_layers = saved_args.get('lstm_layers', LSTM_LAYERS)
    model = SpatioTemporalNetwork(pretrained=False, dropout=dropout,
                                  lstm_layers=lstm_layers)
    # strict=False so that old checkpoints (without attn_pool) load gracefully
    missing, unexpected = model.load_state_dict(
        state['model_state_dict'], strict=False
    )
    if missing:
        print(f'[load_checkpoint] Initialised from scratch: {missing}')
    if unexpected:
        print(f'[load_checkpoint] Ignored unexpected keys: {unexpected}')
    return model
