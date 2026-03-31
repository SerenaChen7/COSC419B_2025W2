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

# Minimum mean per-frame confidence required to make a prediction.
# Mean (rather than sum) is scale-invariant to tracklet length, so a long
# tracklet of weakly-confident frames does not trivially pass the gate.
MEAN_CONFIDENCE_THRESHOLD = 0.45

# Minimum number of valid (post-filter) frames required to make a prediction.
# Protects against tracklets where almost every frame was filtered by the
# entropy/confidence gates in predict.py, leaving only 1-2 noisy votes.
MIN_VOTES = 3

# When 1-digit and 2-digit predictions coexist in a tracklet, multiply
# 1-digit confidences by this factor before voting.
ONE_DIGIT_DOWN_WEIGHT = 0.5

# Minimum number of two-digit predictions required before we start down-
# weighting single-digit predictions.  A single noisy two-digit frame (e.g.
# background text misread as "34") should not suppress genuine #3 or #4 votes.
MIN_TWO_DIGIT_VOTES = 3


def _is_valid_jersey(text: str) -> bool:
    """Return True if text is a valid jersey number string (1-99)."""
    if not text or not text.isdigit():
        return False
    n = int(text)
    return 1 <= n <= 99


def consolidate_tracklet(
    frame_predictions: list[tuple[str, float]],
    mean_confidence_threshold: float = MEAN_CONFIDENCE_THRESHOLD,
    one_digit_down_weight: float = ONE_DIGIT_DOWN_WEIGHT,
    min_votes: int = MIN_VOTES,
    min_two_digit_votes: int = MIN_TWO_DIGIT_VOTES,
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

    mean_confidence_threshold : float
        Minimum *mean* per-frame confidence required to make a prediction.
        Using the mean (rather than the sum) makes the threshold scale-
        invariant to tracklet length: a long tracklet of weakly-confident
        frames cannot trivially pass the gate.  Returns -1 if below threshold.

    one_digit_down_weight : float
        Multiplier applied to 1-digit prediction confidences when the
        tracklet also contains enough 2-digit predictions (see
        min_two_digit_votes).  Reduces the impact of partial-occlusion
        misreads (e.g. "34" -> "3").

    min_votes : int
        Minimum number of valid frames required after filtering.  If fewer
        frames survive, the tracklet is too uncertain to predict and -1 is
        returned.

    min_two_digit_votes : int
        How many 2-digit predictions must be present before we start
        down-weighting single-digit votes.  Prevents a single noisy two-
        digit frame from suppressing genuine single-digit jersey numbers.

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

    # Require a minimum number of contributing frames
    if len(valid) < min_votes:
        return -1

    # Check mean per-frame confidence against threshold
    total_confidence = sum(conf for _, conf in valid)
    mean_confidence = total_confidence / len(valid)
    if mean_confidence < mean_confidence_threshold:
        return -1

    # Only down-weight single-digit votes when enough two-digit predictions
    # exist to be confident that the jersey number is genuinely two digits.
    two_digit_count = sum(1 for num, _ in valid if len(num) == 2)
    has_confident_two_digit = two_digit_count >= min_two_digit_votes

    # Accumulate confidence-weighted votes
    votes: dict[int, float] = defaultdict(float)
    for num_str, conf in valid:
        jersey_num = int(num_str)
        if has_confident_two_digit and len(num_str) == 1:
            conf *= one_digit_down_weight
        votes[jersey_num] += conf

    return max(votes, key=votes.__getitem__)


def consolidate_all(
    tracklet_predictions: dict[str, list[tuple[str, float]]],
    mean_confidence_threshold: float = MEAN_CONFIDENCE_THRESHOLD,
    one_digit_down_weight: float = ONE_DIGIT_DOWN_WEIGHT,
    min_votes: int = MIN_VOTES,
    min_two_digit_votes: int = MIN_TWO_DIGIT_VOTES,
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
        tid: consolidate_tracklet(
            preds, mean_confidence_threshold, one_digit_down_weight,
            min_votes, min_two_digit_votes,
        )
        for tid, preds in tracklet_predictions.items()
    }
