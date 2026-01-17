"""Data structures for the visual pipeline stages.

This module defines frozen dataclasses for face detection, clustering, scene analysis,
and visual descriptions used in the computer vision pipeline.

Adheres to Memoirr coding standards: frozen dataclasses, type hints, Google-style docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class FaceDetection:
    """Represents a detected face in a video frame.
    
    Attributes:
        bbox: Bounding box coordinates as (x, y, width, height) in pixels.
        embedding: 512-dimensional face embedding vector for similarity comparison.
        confidence: Detection confidence score (0.0 to 1.0).
        frame_path: Path to the frame image file containing this face.
    """
    bbox: tuple[int, int, int, int]
    embedding: List[float]
    confidence: float
    frame_path: Path


@dataclass(frozen=True)
class FaceCluster:
    """Represents a cluster of similar faces, potentially belonging to one character.
    
    Attributes:
        cluster_id: Unique identifier for this cluster.
        exemplar_embedding: Centroid embedding vector representing the cluster.
        member_count: Number of face detections in this cluster.
        label: Optional actor name if the cluster has been matched to a cast member.
    """
    cluster_id: int
    exemplar_embedding: List[float]
    member_count: int
    label: Optional[str] = None


@dataclass(frozen=True)
class VisualDescription:
    """Represents an AI-generated description of visual content.
    
    Attributes:
        text: Natural language description of the visual content.
        model_used: Name/identifier of the model that generated the description.
        confidence: Optional confidence score for the description accuracy (0.0 to 1.0).
    """
    text: str
    model_used: str
    confidence: Optional[float] = None


@dataclass(frozen=True)
class Scene:
    """Represents a detected scene segment in a video with associated metadata.
    
    Attributes:
        scene_id: Unique identifier for this scene.
        start_frame: Frame number where the scene begins.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
        keyframe_paths: List of paths to representative keyframes extracted from the scene.
        visual_description: Optional AI-generated description of the scene's visual content.
        detected_faces: List of faces detected within this scene.
        face_clusters: List of cluster IDs for faces found in this scene.
    """
    scene_id: int
    start_frame: int
    start_ms: int
    end_ms: int
    keyframe_paths: List[Path]
    visual_description: Optional[VisualDescription] = None
    detected_faces: List[FaceDetection] = field(default_factory=list)
    face_clusters: List[int] = field(default_factory=list)
