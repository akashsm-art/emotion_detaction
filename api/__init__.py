"""
FastAPI Backend Package for Multi-Modal Emotion Recognition System.
Exposes REST endpoints for individual modalities and confidence-aware late fusion.
"""

from api.main import app

__all__ = ["app"]
