"""
Tracklet-level prediction consolidation.

Implements the heuristic consolidation method from:
    Koshkina & Elder, "A General Framework for Jersey Number Recognition
    in Sports Video", CVPRW 2024.

Given a list of per-frame (predicted_number, confidence) pairs from a
tracklet, this module aggregates them into a single tracklet-level
jersey number prediction.

Algorithm (Section 3.4.4):
    1. Confidence-weighted majority vote over all legible frame predictions.
    2. If any 2-digit predictions exist, down-weight 1-digit predictions
       (partial occlusion often hides one digit of a 2-digit number).
    3. If total confidence is below a threshold, return -1 (illegible).

Usage:
    from consolidate import consolidate_tracklet

    frame_predictions = [
        ("4",  0.92),
        ("4",  0.85),
        ("14", 0.78),
        ("4",  0.60),
        ("1",  0.40),
    ]
    result = consolidate_tracklet(frame_predictions)
    print(result)  # -> 4
"""

from collections import defaultdict


# ---------------------------------------------------------------------------
# Tunable constants (match paper's best-performing configuration)
# ---------------------------------------------------------------------------

# If total confidence across all frames is below this, output -1.
CONFIDENCE_THRESHOLD = 2.0

# When 1-digit and 2-digit predictions coexist in a tracklet, multiply
# 1-digit confidences by this factor before voting.
ONE_DIGIT_DOWN_WEIGHT = 0.5


def _is_valid_jersey(text: str) -> bool:
    """Return True if text is a valid jersey number string (1-99)."""
    if not text or not text.isdigit():
        return False
    n = int(text)
    return 1 <= n <= 99


def consolidate_tracklet(
    frame_predictions: list[tuple[str, float]],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    one_digit_down_weight: float = ONE_DIGIT_DOWN_WEIGHT,
) -> int:
    """
    Aggregate frame-level STR predictions into a single tracklet prediction.

    Parameters
    ----------
    frame_predictions : list of (predicted_number, confidence)
        Each element is a (str, float) pair from one legible frame.
        - predicted_number: raw string output from PARSeq (e.g. "4", "14")
        - confidence: scalar confidence in [0, 1]
        Frames where the legibility classifier returned illegible should
        be excluded before calling this function.

    confidence_threshold : float
        Minimum total confidence required to make a prediction.
        If the sum of all confidences is below this, returns -1.

    one_digit_down_weight : float
        Multiplier applied to 1-digit prediction confidences when the
        tracklet also contains at least one 2-digit prediction.
        Reduces the impact of partial-occlusion misreads (e.g. "34" -> "3").

    Returns
    -------
    int
        Predicted jersey number (1–99), or -1 if the tracklet is illegible
        or no valid prediction could be made.
    """
    # Filter to valid jersey number predictions only
    valid = [
        (num, conf)
        for num, conf in frame_predictions
        if _is_valid_jersey(num) and conf > 0
    ]

    if not valid:
        return -1

    # Check total confidence against threshold
    total_confidence = sum(conf for _, conf in valid)
    if total_confidence < confidence_threshold:
        return -1

    # Determine if any 2-digit predictions exist in this tracklet
    has_two_digit = any(len(num) == 2 for num, _ in valid)

    # Accumulate confidence-weighted votes
    votes: dict[int, float] = defaultdict(float)
    for num_str, conf in valid:
        jersey_num = int(num_str)
        if has_two_digit and len(num_str) == 1:
            conf *= one_digit_down_weight
        votes[jersey_num] += conf

    return max(votes, key=votes.__getitem__)


def consolidate_all(
    tracklet_predictions: dict[str, list[tuple[str, float]]],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    one_digit_down_weight: float = ONE_DIGIT_DOWN_WEIGHT,
) -> dict[str, int]:
    """
    Consolidate predictions for an entire dataset split.

    Parameters
    ----------
    tracklet_predictions : dict
        Maps tracklet_id -> list of (predicted_number, confidence) pairs.

    Returns
    -------
    dict
        Maps tracklet_id -> jersey number int (1-99) or -1.
    """
    return {
        tid: consolidate_tracklet(preds, confidence_threshold, one_digit_down_weight)
        for tid, preds in tracklet_predictions.items()
    }
