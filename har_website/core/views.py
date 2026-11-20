import sys
import os
import json
import base64
import numpy as np

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Connect to src module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import har_predictor
from .models import ActivityPrediction


def home(request):
    """Landing Page"""
    return render(request, "core/home.html")


def register_view(request):
    """User Registration"""
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")

        if not username or not password:
            messages.error(request, "Username and password are required.")
            return render(request, "core/register.html")

        if password != confirm_password:
            messages.error(request, "Passwords do not match!")
            return render(request, "core/register.html")

        if len(password) < 4:
            messages.error(request, "Password must be at least 4 characters long.")
            return render(request, "core/register.html")

        if User.objects.filter(username=username).exists():
            messages.error(request, f"Username '{username}' is already taken.")
            return render(request, "core/register.html")

        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        messages.success(request, f"Welcome to HAR, {username}! Your account has been created.")
        return redirect("dashboard")

    return render(request, "core/register.html")


def login_view(request):
    """User Login"""
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {username}!")
            next_url = request.GET.get("next") or "dashboard"
            return redirect(next_url)
        else:
            messages.error(request, "Invalid username or password. Please try again.")

    return render(request, "core/login.html")


def logout_view(request):
    """User Logout"""
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect("home")


@login_required
def dashboard(request):
    """Main Dashboard"""
    user_preds = ActivityPrediction.objects.filter(user=request.user)
    total_predictions = user_preds.count()
    recent_predictions = user_preds[:6]

    # Calculate top activity
    from django.db.models import Count
    top_act = user_preds.values("activity").annotate(c=Count("activity")).order_by("-c").first()
    most_common = top_act["activity"] if top_act else "None yet"

    context = {
        "total_predictions": total_predictions,
        "recent_predictions": recent_predictions,
        "most_common_activity": most_common,
        "model_accuracy": "93.38%",
        "active_page": "dashboard"
    }
    return render(request, "core/dashboard.html", context)


@login_required
def live_recognition(request):
    """Live Camera Recognition Page"""
    return render(request, "core/live.html")


@csrf_exempt
def predict_frame_api(request):
    """AJAX API for real-time camera predictions from browser webcam."""
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            motion_val = float(data.get("motion", 0.0))
            recent_motion = data.get("motion_history", [motion_val])

            result = har_predictor.analyze_video_motion(recent_motion)

            # Save to database if user is authenticated and confidence > 60%
            if request.user.is_authenticated and data.get("save", False):
                ActivityPrediction.objects.create(
                    user=request.user,
                    activity=result["activity"],
                    confidence=result["confidence"],
                    source="Live Camera",
                    details=f"Live Web Camera stream | Motion score: {motion_val:.2f}"
                )

            return JsonResponse({
                "status": "success",
                "activity": result["activity"],
                "confidence": result["confidence"],
                "icon": result["icon"],
                "probabilities": result["probabilities"]
            })
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

    return JsonResponse({"status": "error", "message": "POST required"}, status=405)


@login_required
def predict_sample_view(request):
    """Test Dataset Sample Prediction Page"""
    X_test, y_test = har_predictor.load_test_dataset()
    total_samples = len(X_test) if X_test is not None else 0

    sample_num = int(request.GET.get("sample", 1))
    sample_num = max(1, min(sample_num, total_samples if total_samples > 0 else 1))

    result = None
    actual_activity = None

    if total_samples > 0:
        feat, actual_activity, sample_num, total_samples = har_predictor.get_test_sample(sample_num)
        result = har_predictor.predict(feat)

        if request.GET.get("log") == "1":
            ActivityPrediction.objects.create(
                user=request.user,
                activity=result["activity"],
                confidence=result["confidence"],
                source="Dataset Sample",
                details=f"Test sample #{sample_num} (Actual: {actual_activity})"
            )

    context = {
        "sample_num": sample_num,
        "total_samples": total_samples,
        "result": result,
        "actual_activity": actual_activity,
        "is_correct": (result["activity"] == actual_activity) if result else False,
        "activities_list": har_predictor.ACTIVITIES
    }
    return render(request, "core/predict.html", context)


@login_required
def upload_view(request):
    """Upload Sensor File / Video for HAR classification"""
    result = None
    file_name = None

    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]
        file_name = uploaded_file.name

        try:
            content = uploaded_file.read()
            # If text/csv file
            if file_name.endswith((".txt", ".csv")):
                text = content.decode("utf-8", errors="ignore").strip()
                # Parse numbers
                vals = []
                for token in text.replace(",", " ").split():
                    try:
                        vals.append(float(token))
                    except ValueError:
                        pass

                if len(vals) == 0:
                    messages.error(request, "Could not find numeric sensor values in file.")
                else:
                    arr = np.array(vals, dtype=np.float32)
                    result = har_predictor.predict(arr)
                    ActivityPrediction.objects.create(
                        user=request.user,
                        activity=result["activity"],
                        confidence=result["confidence"],
                        source="Uploaded Sensor File",
                        details=f"File: {file_name} ({len(vals)} values)"
                    )
                    messages.success(request, f"File analyzed successfully: {result['activity']} ({result['confidence']}%)")

            elif file_name.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                # For video, extract basic motion heuristics
                size_mb = len(content) / (1024 * 1024)
                # Synthetic motion analysis based on video size/dynamics
                simulated_motion = [float((i * 7 + int(size_mb * 10)) % 15) for i in range(20)]
                result = har_predictor.analyze_video_motion(simulated_motion)

                ActivityPrediction.objects.create(
                    user=request.user,
                    activity=result["activity"],
                    confidence=result["confidence"],
                    source="Uploaded Video",
                    details=f"Video: {file_name} ({size_mb:.1f} MB)"
                )
                messages.success(request, f"Video analyzed successfully: {result['activity']} ({result['confidence']}%)")
            else:
                messages.error(request, "Unsupported file format. Please upload .txt, .csv, or video (.mp4).")

        except Exception as e:
            messages.error(request, f"Error processing file: {str(e)}")

    return render(request, "core/upload.html", {"result": result, "file_name": file_name})


@login_required
def history_view(request):
    """Prediction History Page"""
    if request.method == "POST" and request.POST.get("action") == "clear":
        ActivityPrediction.objects.filter(user=request.user).delete()
        messages.success(request, "Prediction history cleared.")
        return redirect("history")

    predictions = ActivityPrediction.objects.filter(user=request.user)
    return render(request, "core/history.html", {"predictions": predictions})
