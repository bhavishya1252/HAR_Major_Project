"""
Human Activity Recognition - CLI Application
Interactive console tool to test HAR model predictions on dataset samples.
"""

import sys
import os
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import har_predictor


def print_banner():
    print("=" * 55)
    print("      HUMAN ACTIVITY RECOGNITION (HAR) SYSTEM")
    print("   Deep Learning Model (93.38% Test Accuracy)")
    print("=" * 55)


def run_sample_prediction(sample_num=1):
    feat, actual, num, total = har_predictor.get_test_sample(sample_num)
    if feat is None:
        print("Dataset not found at dataset/UCI_HAR/test.")
        return

    result = har_predictor.predict(feat)

    print("\n" + "-" * 40)
    print(f" Sample Number       : {num} / {total}")
    print(f" Actual Activity     : {actual}")
    print(f" Predicted Activity  : {result['activity']}")
    print(f" Confidence          : {result['confidence']}%")
    print(f" Match               : {'CORRECT' if result['activity'] == actual else 'MISMATCH'}")
    print("-" * 40)
    print("\n Class Probabilities:")
    for act, prob in result['probabilities'].items():
        bar = "#" * int(prob / 5)
        print(f"   {act:<20} {prob:>6.2f}% | {bar}")
    print("-" * 40)


def main():
    print_banner()

    X_test, y_test = har_predictor.load_test_dataset()
    total_samples = len(X_test) if X_test is not None else 0
    print(f"\nModel and Test Dataset loaded ({total_samples} samples available).")

    while True:
        print("\nChoose an option:")
        print(" 1. Predict random test sample")
        print(" 2. Enter specific sample number (1 to", total_samples, ")")
        print(" 3. Test first 10 samples (Accuracy check)")
        print(" 4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            import random
            rand_idx = random.randint(1, total_samples) if total_samples > 0 else 1
            run_sample_prediction(rand_idx)

        elif choice == "2":
            val = input(f"Enter sample number (1-{total_samples}): ").strip()
            try:
                s_num = int(val)
                if 1 <= s_num <= total_samples:
                    run_sample_prediction(s_num)
                else:
                    print(f"Please enter a number between 1 and {total_samples}.")
            except ValueError:
                print("Invalid number.")

        elif choice == "3":
            print("\nEvaluating first 10 samples...")
            correct = 0
            for i in range(1, 11):
                feat, actual, num, _ = har_predictor.get_test_sample(i)
                res = har_predictor.predict(feat)
                is_ok = (res['activity'] == actual)
                if is_ok:
                    correct += 1
                status = "OK" if is_ok else "X"
                print(f" Sample {i:2d}: Actual={actual:<18} Pred={res['activity']:<18} ({res['confidence']:5.1f}%) [{status}]")
            print(f"\nResult: {correct}/10 correct ({correct * 10}% accuracy).")

        elif choice == "4":
            print("\nHAR System closed. Goodbye!")
            break

        else:
            print("Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()
