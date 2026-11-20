"""
Human Activity Recognition - Test Single Prediction
Quickly evaluates model on test samples.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import har_predictor

feat, actual, num, total = har_predictor.get_test_sample(1)
if feat is not None:
    res = har_predictor.predict(feat)
    print("=" * 45)
    print("   HAR Model Prediction Test")
    print("=" * 45)
    print(f"Sample       : #{num} of {total}")
    print(f"Actual       : {actual}")
    print(f"Predicted    : {res['activity']}")
    print(f"Confidence   : {res['confidence']}%")
    print(f"Result       : {'CORRECT' if res['activity'] == actual else 'MISMATCH'}")
    print("=" * 45)
else:
    print("Error: Could not load test dataset.")
