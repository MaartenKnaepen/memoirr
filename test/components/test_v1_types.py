"""Tests for V1 type definitions (metadata and vision).

Tests frozen dataclasses, default values, and immutability for the types
defined in src/components/metadata/utilities/types.py and
src/components/vision/utilities/types.py.

Adheres to Memoirr coding standards: type hints, Google-style docstrings.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import List

import pytest

from src.components.metadata.utilities.types import CastMember, MovieMetadata
from src.components.vision.utilities.types import (
    FaceCluster,
    FaceDetection,
    Scene,
    VisualDescription,
)


class TestMetadataTypes:
    """Tests for metadata type definitions."""

    def test_cast_member_creation(self) -> None:
        """Test basic CastMember creation with all fields."""
        cast = CastMember(
            name="Ian McKellen",
            character="Gandalf",
            tmdb_id=1327,
            profile_path="/path/to/image.jpg"
        )
        
        assert cast.name == "Ian McKellen"
        assert cast.character == "Gandalf"
        assert cast.tmdb_id == 1327
        assert cast.profile_path == "/path/to/image.jpg"

    def test_cast_member_defaults(self) -> None:
        """Test CastMember with optional profile_path as None."""
        cast = CastMember(
            name="Elijah Wood",
            character="Frodo Baggins",
            tmdb_id=109,
            profile_path=None
        )
        
        assert cast.profile_path is None

    def test_cast_member_immutability(self) -> None:
        """Test that CastMember is frozen and cannot be modified."""
        cast = CastMember(
            name="Viggo Mortensen",
            character="Aragorn",
            tmdb_id=110,
        )
        
        with pytest.raises(FrozenInstanceError):
            cast.name = "Someone Else"  # type: ignore[misc]

    def test_movie_metadata_creation(self) -> None:
        """Test MovieMetadata creation with all fields."""
        cast_members: List[CastMember] = [
            CastMember(name="Actor 1", character="Character 1", tmdb_id=1),
            CastMember(name="Actor 2", character="Character 2", tmdb_id=2),
        ]
        
        movie = MovieMetadata(
            title="The Lord of the Rings: The Fellowship of the Ring",
            year=2001,
            tmdb_id=120,
            radarr_id=1,
            plex_rating_key="12345",
            cast=cast_members,
            genres=["Adventure", "Fantasy", "Action"],
            overview="A meek Hobbit and companions set out on a journey..."
        )
        
        assert movie.title == "The Lord of the Rings: The Fellowship of the Ring"
        assert movie.year == 2001
        assert movie.tmdb_id == 120
        assert movie.radarr_id == 1
        assert movie.plex_rating_key == "12345"
        assert len(movie.cast) == 2
        assert movie.genres == ["Adventure", "Fantasy", "Action"]
        assert "Hobbit" in movie.overview  # type: ignore[operator]

    def test_movie_metadata_defaults(self) -> None:
        """Test MovieMetadata with default empty lists and None values."""
        movie = MovieMetadata(
            title="Test Movie",
            year=2020,
            tmdb_id=999,
            radarr_id=None,
            plex_rating_key=None,
            cast=[]
        )
        
        assert movie.radarr_id is None
        assert movie.plex_rating_key is None
        assert movie.cast == []
        assert movie.genres == []
        assert movie.overview is None

    def test_movie_metadata_immutability(self) -> None:
        """Test that MovieMetadata is frozen and cannot be modified."""
        movie = MovieMetadata(
            title="Test Movie",
            year=2020,
            tmdb_id=999,
            radarr_id=None,
            plex_rating_key=None,
            cast=[]
        )
        
        with pytest.raises(FrozenInstanceError):
            movie.title = "Changed Title"  # type: ignore[misc]


class TestVisionTypes:
    """Tests for vision type definitions."""

    def test_face_detection_creation(self) -> None:
        """Test FaceDetection creation with all fields."""
        embedding = [0.1] * 512  # 512-dimensional vector
        face = FaceDetection(
            bbox=(100, 150, 200, 250),
            embedding=embedding,
            confidence=0.95,
            frame_path=Path("/frames/frame_001.jpg")
        )
        
        assert face.bbox == (100, 150, 200, 250)
        assert len(face.embedding) == 512
        assert face.confidence == 0.95
        assert face.frame_path == Path("/frames/frame_001.jpg")

    def test_face_detection_immutability(self) -> None:
        """Test that FaceDetection is frozen and cannot be modified."""
        face = FaceDetection(
            bbox=(10, 20, 30, 40),
            embedding=[0.1] * 512,
            confidence=0.9,
            frame_path=Path("/test.jpg")
        )
        
        with pytest.raises(FrozenInstanceError):
            face.confidence = 0.8  # type: ignore[misc]

    def test_face_cluster_creation(self) -> None:
        """Test FaceCluster creation with all fields."""
        cluster = FaceCluster(
            cluster_id=1,
            exemplar_embedding=[0.2] * 512,
            member_count=15,
            label="Gandalf"
        )
        
        assert cluster.cluster_id == 1
        assert len(cluster.exemplar_embedding) == 512
        assert cluster.member_count == 15
        assert cluster.label == "Gandalf"

    def test_face_cluster_defaults(self) -> None:
        """Test FaceCluster with optional label as None."""
        cluster = FaceCluster(
            cluster_id=2,
            exemplar_embedding=[0.3] * 512,
            member_count=8,
            label=None
        )
        
        assert cluster.label is None

    def test_face_cluster_immutability(self) -> None:
        """Test that FaceCluster is frozen and cannot be modified."""
        cluster = FaceCluster(
            cluster_id=1,
            exemplar_embedding=[0.1] * 512,
            member_count=10
        )
        
        with pytest.raises(FrozenInstanceError):
            cluster.label = "New Label"  # type: ignore[misc]

    def test_visual_description_creation(self) -> None:
        """Test VisualDescription creation with all fields."""
        desc = VisualDescription(
            text="A dark cave with glowing crystals",
            model_used="gpt-4-vision",
            confidence=0.88
        )
        
        assert desc.text == "A dark cave with glowing crystals"
        assert desc.model_used == "gpt-4-vision"
        assert desc.confidence == 0.88

    def test_visual_description_defaults(self) -> None:
        """Test VisualDescription with optional confidence as None."""
        desc = VisualDescription(
            text="A battlefield scene",
            model_used="llava",
            confidence=None
        )
        
        assert desc.confidence is None

    def test_visual_description_immutability(self) -> None:
        """Test that VisualDescription is frozen and cannot be modified."""
        desc = VisualDescription(
            text="Test description",
            model_used="test-model"
        )
        
        with pytest.raises(FrozenInstanceError):
            desc.text = "Changed description"  # type: ignore[misc]

    def test_scene_creation(self) -> None:
        """Test Scene creation with all fields."""
        faces: List[FaceDetection] = [
            FaceDetection(
                bbox=(10, 20, 30, 40),
                embedding=[0.1] * 512,
                confidence=0.9,
                frame_path=Path("/frame1.jpg")
            )
        ]
        
        visual_desc = VisualDescription(
            text="Epic battle scene",
            model_used="gpt-4-vision"
        )
        
        scene = Scene(
            scene_id=1,
            start_frame=0,
            start_ms=0,
            end_ms=5000,
            keyframe_paths=[Path("/keyframe1.jpg"), Path("/keyframe2.jpg")],
            visual_description=visual_desc,
            detected_faces=faces,
            face_clusters=[1, 2, 3]
        )
        
        assert scene.scene_id == 1
        assert scene.start_frame == 0
        assert scene.start_ms == 0
        assert scene.end_ms == 5000
        assert len(scene.keyframe_paths) == 2
        assert scene.visual_description is not None
        assert scene.visual_description.text == "Epic battle scene"
        assert len(scene.detected_faces) == 1
        assert scene.face_clusters == [1, 2, 3]

    def test_scene_defaults(self) -> None:
        """Test Scene with default empty lists and None values."""
        scene = Scene(
            scene_id=2,
            start_frame=100,
            start_ms=10000,
            end_ms=15000,
            keyframe_paths=[Path("/keyframe.jpg")]
        )
        
        assert scene.visual_description is None
        assert scene.detected_faces == []
        assert scene.face_clusters == []

    def test_scene_immutability(self) -> None:
        """Test that Scene is frozen and cannot be modified."""
        scene = Scene(
            scene_id=1,
            start_frame=0,
            start_ms=0,
            end_ms=1000,
            keyframe_paths=[Path("/test.jpg")]
        )
        
        with pytest.raises(FrozenInstanceError):
            scene.scene_id = 999  # type: ignore[misc]
