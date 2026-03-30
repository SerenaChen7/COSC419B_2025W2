"""
Tracklet-level prediction consolidation.

Implements the heuristic consolidation from:
    Koshkina & Elder, "A General Framework for Jersey Number Recognition
    in Sports Video", CVPRW 2024.

Given per-frame (predicted_number_str, confidence) pairs, aggregates them
into a single tracklet-level jersey number prediction.

Algorithm:
    1. Confidence-weighted majority vote over all legible frame predictions.
    2. If any 2-digit predictions exist, down-weight 1-digit predictions
       (partial occlusion often hides one digit of a 2-digit number).
    3. If total confidence is below a threshold, return -1 (illegible).
"""
from collections import defaultdict

CONFIDENCE_THRESHOLD  = 0.3   # min total confidence to make a prediction
ONE_DIGIT_DOWN_WEIGHT = 0.5   # penalise 1-digit reads when 2-digit reads exist


def _valid(text: str) -> bool:
    return bool(text) and text.isdigit() and 1 <= int(text) <= 99


def consolidate_tracklet(
    frame_predictions: list[tuple[str, float]],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    one_digit_down_weight: float = ONE_DIGIT_DOWN_WEIGHT,
) -> int:
    """
    Aggregate frame-level predictions into one tracklet-level jersey number.

    Parameters
    ----------
    frame_predictions : list of (number_string, confidence)
        Only include legible frames (exclude frames predicted as illegible).

    Returns
    -------
    int : jersey number 1-99, or -1 if illegible / low confidence.
    """
    valid = [(num, conf) for num, conf in frame_predictions if _valid(num) and conf > 0]
    if not valid:
        return -1

    if sum(c for _, c in valid) < confidence_threshold:
        return -1

    has_two_digit = any(len(num) == 2 for num, _ in valid)
    votes: dict[int, float] = defaultdict(float)
    for num_str, conf in valid:
        if has_two_digit and len(num_str) == 1:
            conf *= one_digit_down_weight
        votes[int(num_str)] += conf

    return max(votes, key=votes.__getitem__)
