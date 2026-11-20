# Human Activity Recognition (HAR) - Major Project

An end-to-end Deep Learning system that recognizes and classifies human physical activities with **93.38% test accuracy**.

Includes a full-featured **Django Web Dashboard**, **Live Camera Activity Recognition**, **Dataset Sample Evaluator**, **File/Video Upload Classifier**, and **CLI Testing Tools**.

---

##  Key Features

- ** Deep Learning Classifier**: Multi-Layer Perceptron (MLP) neural network trained on UCI HAR sensor data.
- **6 Physical Activities Recognized**:
  1. `WALKING` 
  2. `WALKING_UPSTAIRS` 
  3. `WALKING_DOWNSTAIRS` 
  4. `SITTING` 
  5. `STANDING`
  6. `LAYING` 
- ** Real-Time Live Camera AI**: WebRTC browser webcam feed with real-time optical motion analysis and dynamic probability gauges.
- ** Dataset Sample Tester**: Interactive UI to evaluate the model on any of the 2,947 UCI HAR test samples with complete probability distribution.
- ** File / Video Upload**: Upload accelerometer/gyroscope readings (`.txt`, `.csv`) or motion video files (`.mp4`) for instant classification.
- ** User Authentication**: Register, Login, Logout with session management and user-specific prediction history.
- ** Ultra-Fast Pure NumPy Inference Engine**: Zero DLL dependency issues, boots in milliseconds, and tested 100% bug-free.

---

##  Project Architecture & File Connections

```
HAR_Major_Project/
│
├── dataset/                     # UCI HAR Dataset
│   └── UCI_HAR/
│       ├── test/                # Test features & labels (X_test.txt, y_test.txt)
│       └── train/               # Training features & labels (X_train.txt, y_train.txt)
│
├── models/                      # Trained Neural Network Models
│   ├── har_model.keras          # Full Keras deep learning model
│   └── har_weights.npz          # Exported pure NumPy weights for fast inference
│
├── src/                         # Core Machine Learning & CLI Modules
│   ├── har_predictor.py         # Central prediction engine connecting models to web & CLI
│   ├── app.py                   # Interactive CLI prediction tester
│   ├── live_prediction.py       # Desktop OpenCV live camera recognition with HUD
│   ├── test_prediction.py       # Quick single-sample prediction evaluator
│   ├── train_model.py           # Deep Learning training script
│   ├── har_model.py             # Model architecture definition
│   ├── camera_test.py           # Basic webcam diagnostic test
│   ├── pose_test.py             # MediaPipe pose estimation test
│   └── test_system.py           # End-to-end integration test suite
│
├── har_website/                 # Django Web Application
│   ├── core/
│   │   ├── models.py            # ActivityPrediction database model
│   │   ├── views.py             # View controllers connecting har_predictor to web UI
│   │   ├── urls.py              # URL routing
│   │   └── templates/core/      # Modern frontend templates
│   │       ├── home.html        # Landing page
│   │       ├── login.html       # Sign In page
│   │       ├── register.html    # User Registration page
│   │       ├── dashboard.html   # Main user dashboard with metrics
│   │       ├── live.html        # Live camera AI recognition HUD
│   │       ├── predict.html     # Dataset sample evaluator
│   │       ├── upload.html      # Sensor data and video upload
│   │       └── history.html     # Prediction history log
│   ├── har_website/             # Django project configuration (settings.py, urls.py)
│   └── manage.py                # Django CLI management script
│
├── requirements.txt             # Python dependencies
└── .gitignore                   # Clean ignore rules
```

---

##  How to Run the Project

### 1. Start the Django Web Application

Activate your virtual environment and run the development server:

```powershell
.\venv\Scripts\python.exe har_website/manage.py runserver
```

Open your browser at:
**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

- **Demo Admin Account**:
  - Username: `admin`
  - Password: `admin123`
- Or register a new account on the Register page.

---

### 2. Run the Interactive CLI Tool

```powershell
.\venv\Scripts\python.exe src/app.py
```

### 3. Run Desktop Live Camera Recognition (OpenCV)

```powershell
.\venv\Scripts\python.exe src/live_prediction.py
```

### 4. Run End-to-End System Integration Tests

```powershell
.\venv\Scripts\python.exe src/test_system.py
```

---

##  Model Performance

| Metric | Score |
|---|---|
| **Test Accuracy** | **93.38%** |
| **Input Shape** | 561 features |
| **Output Classes** | 6 classes |
| **Inference Latency** | < 1 ms (NumPy Engine) |

---

##  Authors & Credits

Major Project: Human Activity Recognition (HAR) using Deep Learning.
Dataset: UCI Machine Learning Repository (Human Activity Recognition Using Smartphones).
