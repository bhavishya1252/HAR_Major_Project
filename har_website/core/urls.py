from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("live/", views.live_recognition, name="live"),
    path("api/predict_frame/", views.predict_frame_api, name="predict_frame_api"),
    path("predict/", views.predict_sample_view, name="predict_sample"),
    path("upload/", views.upload_view, name="upload"),
    path("history/", views.history_view, name="history"),
]
