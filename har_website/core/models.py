from django.db import models
from django.contrib.auth.models import User


class ActivityPrediction(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="predictions"
    )
    activity = models.CharField(max_length=60)
    confidence = models.FloatField(default=0.0)
    source = models.CharField(max_length=60, default="Dataset Sample")
    details = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.activity} ({self.confidence:.1f}%) - {self.source}"
