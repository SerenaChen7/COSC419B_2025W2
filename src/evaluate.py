import json
import argparse
import sys


def evaluate(prediction_path, ground_truth_path):
    # 1. Load Ground Truth
    try:
        with open(ground_truth_path, 'r') as f:
            gt_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)

    # 2. Load Student Predictions
    try:
        with open(prediction_path, 'r') as f:
            pred_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {prediction_path}")
        sys.exit(1)

    correct = 0
    total = 0
    missing = 0

    # 3. Compare
    for uuid, gt_number in gt_data.items():
        total += 1
        # Check if the student made a prediction for this UUID
        if uuid in pred_data:
            pred_number = pred_data[uuid]
            # Ensure we compare integers (handle potential string inputs)
            try:
                if int(pred_number) == int(gt_number):
                    correct += 1
            except ValueError:
                # If prediction is not a valid number, it counts as incorrect
                pass
        else:
            missing += 1

    # 4. Results
    accuracy = correct / total if total > 0 else 0
    print("-" * 30)
    print("EVALUATION RESULTS")
    print("-" * 30)
    print(f"Total Samples (GT): {total}")
    print(f"Predictions Provided: {total - missing}")
    print(f"Correct Predictions: {correct}")
    print(f"Missing Predictions: {missing}")
    print("-" * 30)
    print(f"FINAL ACCURACY: {accuracy:.2%}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Jersey Number Recognition")
    parser.add_argument("--pred", required=True, help="Path to your predictions JSON file")
    parser.add_argument("--gt", required=True, help="Path to the test_gt.json file")
    args = parser.parse_args()
    evaluate(args.pred, args.gt)
