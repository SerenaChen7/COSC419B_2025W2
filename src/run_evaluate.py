"""
Evaluate predictions against ground truth.

Usage:
    python src/run_evaluate.py [--predictions PATH] [--ground-truth PATH]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--predictions',   default='outputs/predictions.json')
parser.add_argument('--ground-truth',  default='data/jersey-2023/test/test_gt.json')
args = parser.parse_args()

evaluate(args.predictions, args.ground_truth)
