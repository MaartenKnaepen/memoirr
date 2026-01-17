"""Vision component package.

This package contains components and utilities for computer vision tasks
including scene detection, face detection, face clustering, and visual descriptions.
"""

from src.components.vision.utilities.types import (
    FaceCluster,
    FaceDetection,
    Scene,
    VisualDescription,
)

__all__ = ["FaceDetection", "FaceCluster", "VisualDescription", "Scene"]
