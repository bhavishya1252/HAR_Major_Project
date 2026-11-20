"""
Human Activity Recognition - Live Camera Recognition
Real-time webcam activity classification using motion features and HAR model.
"""

import sys
import os
import time
import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import har_predictor


def main():
    print("=" * 50)
    print("   HAR - Real-time Camera Activity Recognition")
    print("   Press 'q' in the camera window to exit")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Tip: Ensure camera is plugged in and permissions are enabled.")
        return

    print("Camera initialized successfully. Streaming...")

    prev_gray = None
    motion_history = []
    current_activity = "INITIALIZING"
    current_conf = 0.0
    last_pred_time = time.time()

    # Preload a sample feature for static baseline
    X_test, y_test = har_predictor.load_test_dataset()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_score = 0.0
        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = float(np.sum(thresh) / (w * h))
            motion_history.append(motion_score)
            if len(motion_history) > 30:
                motion_history.pop(0)

        prev_gray = gray

        # Predict every 0.3 seconds based on motion dynamics
        if time.time() - last_pred_time > 0.3 and len(motion_history) > 5:
            last_pred_time = time.time()
            avg_motion = float(np.mean(motion_history))

            res = har_predictor.analyze_video_motion(motion_history)
            current_activity = res["activity"]
            current_conf = res["confidence"]

        # Draw HUD on frame
        # Dark overlay bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(
            frame,
            "HUMAN ACTIVITY RECOGNITION (AI LIVE)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (56, 189, 248),
            2
        )

        # Prediction text
        color = (34, 197, 94) if current_conf > 70 else (234, 179, 8)
        text = f"Activity: {current_activity}  ({current_conf:.1f}%)"
        cv2.putText(
            frame,
            text,
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2
        )

        # Motion indicator
        cv2.putText(
            frame,
            f"Motion Level: {motion_score:.2f} | Press 'q' to quit",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (203, 213, 225),
            1
        )

        cv2.imshow("HAR - Live Activity Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera stream closed.")


if __name__ == "__main__":
    main()
