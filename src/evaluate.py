"""
Evaluate predictions against ground truth.

Usage:
    python src/evaluate.py --pred outputs/predictions.json \
                           --gt data/jersey-2023/test/test_gt.json
"""
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True, help='Path to predictions JSON')
    p.add_argument('--gt', required=True, help='Path to ground truth JSON')
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.pred) as f:
        predictions = json.load(f)
    with open(args.gt) as f:
        gt = json.load(f)

    total = correct = 0
    legible_total = legible_correct = 0
    illegible_total = illegible_correct = 0

    for tid, gt_num in gt.items():
        if tid not in predictions:
            continue
        pred_num = predictions[tid]
        total += 1
        correct += pred_num == gt_num

        if gt_num == -1:
            illegible_total += 1
            illegible_correct += pred_num == -1
        else:
            legible_total += 1
            legible_correct += pred_num == gt_num

    print('=== Evaluation ===')
    print(f'Overall accuracy:    {correct}/{total} = {100*correct/total:.1f}%')
    if legible_total:
        print(f'Legible accuracy:    {legible_correct}/{legible_total} = {100*legible_correct/legible_total:.1f}%')
    if illegible_total:
        print(f'Illegible accuracy:  {illegible_correct}/{illegible_total} = {100*illegible_correct/illegible_total:.1f}%')
    print()
    print(f'Predicted -1 (illegible): {sum(1 for v in predictions.values() if v == -1)} / {len(predictions)}')
    print(f'GT -1 (illegible):        {sum(1 for v in gt.values() if v == -1)} / {len(gt)}')


if __name__ == '__main__':
    main()
