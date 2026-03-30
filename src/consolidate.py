"""
Tracklet-level prediction consolidation via Bayesian log-prob aggregation.

Based on the approach from:
    Koshkina & Elder, "A General Framework for Jersey Number Recognition
    in Sports Video", CVPRW 2024.

Instead of a confidence-weighted vote over per-frame argmax predictions,
we accumulate the full softmax probability vector for every frame and sum
their log-probabilities. This retains all information from the classifier
(not just the top-1 class) and naturally up-weights frames where the model
is more certain, without discarding second-best hypotheses.

Interface
---------
consolidate_tracklet(frame_probs) -> int
    frame_probs : list of array-like, each of shape (NUM_CLASSES,)
                  i.e. the raw softmax output for one frame.
    returns     : jersey number 1-99, or -1 if illegible / no predictions.
"""
import numpy as np

NUM_CLASSES = 100   # class 0 = illegible, class 1-99 = jersey number
MIN_FRAMES  = 1     # require at least this many frames with a legible vote


def consolidate_tracklet(frame_probs: list) -> int:
    """
    Aggregate per-frame softmax probability vectors into one jersey number.

    Parameters
    ----------
    frame_probs : list of array-like, shape (NUM_CLASSES,)
        Full softmax probability vector for each frame (including class 0).

    Returns
    -------
    int : jersey number 1-99, or -1 if the tracklet is illegible.
    """
    if not frame_probs:
        return -1

    probs = np.stack([np.asarray(p, dtype=np.float32) for p in frame_probs])
    # Sum log-probabilities across frames (Bayesian aggregation)
    log_sum = np.sum(np.log(np.clip(probs, 1e-10, 1.0)), axis=0)  # (NUM_CLASSES,)

    # If illegible class dominates after aggregation, return -1
    pred_class = int(np.argmax(log_sum))
    if pred_class == 0:
        return -1

    return pred_class
