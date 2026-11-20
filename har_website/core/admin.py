from django.contrib import admin
from .models import ActivityPrediction


@admin.register(ActivityPrediction)
class ActivityPredictionAdmin(admin.ModelAdmin):
    list_display = ("activity", "confidence", "source", "user", "created_at")
    list_filter = ("activity", "source", "created_at")
    search_fields = ("activity", "source", "user__username")
