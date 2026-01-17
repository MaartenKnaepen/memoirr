Here is the **Detailed Technical Specification** for the V1 Multimodal RAG system.

I have written this as a formal architecture document. Following the document, I have included **LLM 2's review** to confirm we are aligned on the low-level details.

---

### 📄 Document: `docs/specs/v1_multimodal_architecture.md`

# Memoirr V1 Technical Specification: Multimodal Laptop RAG

**Version:** 1.0
**Target Hardware:** Single Node (RTX 4060, 8GB VRAM)
**Execution Strategy:** Sequential Pipeline (Memory Safe)

## 1. Data Schema (The "Scene Node")

We will use **Qdrant** with **Named Vectors** to support hybrid multimodal search without complex fusion logic outside the DB.

### 1.1 Qdrant Collection Config
*   **Collection Name:** `memoirr_v1`
*   **Vector Config:**
    *   `text`: 384 dimensions (SentenceTransformers `all-MiniLM-L6-v2`) - *Distance: Cosine*
    *   `visual`: 384 dimensions (SentenceTransformers `all-MiniLM-L6-v2`) - *Distance: Cosine*
    *   *Note: We embed the visual description TEXT, not the image pixels, to save VRAM and storage.*

### 1.2 Payload Schema
```json
{
  "id": "uuid_v4",
  "payload": {
    "source_file": "The.Two.Towers.2002.mkv",
    "movie_title": "The Lord of the Rings: The Two Towers",
    "tmdb_id": 122,
    "plex_rating_key": "10452", 
    "scene_id": 42,
    "start_ms": 5400000,
    "end_ms": 5450000,
    "duration_ms": 50000,
    
    // Searchable Metadata
    "speakers": ["Gandalf", "Aragorn"],
    "faces_present": ["Ian McKellen", "Viggo Mortensen"],
    "visual_keywords": ["sunrise", "horse", "white_robes", "hill"],
    
    // Content for RAG Context
    "dialogue_text": "Look to my coming on the first light...",
    "visual_description": "Gandalf sits atop a white horse on a steep ridge..."
  },
  "vectors": {
    "text": [0.01, 0.02, ...],
    "visual": [0.05, 0.09, ...] 
  }
}
```

---

## 2. Pipeline Components Specification

### 2.1 Component: `MetadataService`
*   **Inputs:** File path.
*   **Dependencies:** `radarr` (API), `tmdb` (API).
*   **Logic:**
    1.  Parse filename/folder name.
    2.  Query Radarr for `tmdbId` and `plexId` (if Radarr is synced with Plex) OR filename fuzzy match.
    3.  Query TMDB for `credits` (Cast) and `images` (Profiles).
*   **Output:** `MovieMetadata` object (Cast List with Actor Names + Character Names).

### 2.2 Component: `SceneSegmenter`
*   **Inputs:** Video file path.
*   **Dependencies:** `PySceneDetect`, `ffmpeg`.
*   **Logic:**
    1.  Run `detect-content` (threshold=27.0).
    2.  Merge shots shorter than 1.5s into previous shot.
    3.  Extract **3 frames** per scene (Start+1s, Middle, End-1s).
    4.  Save frames to temp dir: `/tmp/memoirr/{movie_id}/frames/scene_{id}_{idx}.jpg`.
*   **Output:** List of `Scene` objects (start/end times, image paths).

### 2.3 Component: `FaceIdentifier` (GPU Pass 1)
*   **Inputs:** List of Frame paths, Cast List (from Metadata).
*   **Dependencies:** `insightface`, `scikit-learn` (DBSCAN).
*   **Model:** `buffalo_l` (640px input).
*   **VRAM Usage:** ~1.2 GB.
*   **Logic:**
    1.  **Batch Detect:** Process frames to extract 512d face embeddings.
    2.  **Cluster:** Global DBSCAN on all movie faces.
    3.  **Label:** 
        *   Load TMDB Actor Headshots.
        *   Compute Cosine Similarity between Cluster Centroids and Actor Headshots.
        *   If sim > 0.6, label Cluster as "Actor Name". Else "Unknown_Cluster_X".
*   **Output:** Mapping `SceneID -> List[ActorNames]`.
*   **Cleanup:** `torch.cuda.empty_cache()`.

### 2.4 Component: `VisualDescriber` (GPU Pass 2)
*   **Inputs:** List of Frame paths, Face Mapping.
*   **Dependencies:** `transformers` (AutoModelForCausalLM).
*   **Model:** `microsoft/Florence-2-large` (float16).
*   **VRAM Usage:** ~2.5 GB (FP16).
*   **Logic:**
    1.  Construct Prompt: `<MORE_DETAILED_CAPTION>`
    2.  *Context Enhancement:* If FaceID says "Ian McKellen" is in frame, append " featuring Ian McKellen" to the resulting caption in post-processing (Florence doesn't take text context well, better to append).
    3.  Generate caption.
*   **Output:** Mapping `SceneID -> VisualText`.
*   **Cleanup:** `torch.cuda.empty_cache()`.

### 2.5 Component: `SpeakerTagger` (Cloud/CPU)
*   **Inputs:** Subtitle file (SRT), Cast List.
*   **Dependencies:** `groq` (API).
*   **Model:** `llama-3-8b-8192` (via Groq).
*   **Logic:**
    1.  Sliding window (20 lines).
    2.  Prompt: *"Assign speakers to these lines based on this Cast List. Return JSON."*
*   **Output:** `EnrichedSRT` (Line -> Speaker).

### 2.6 Component: `FusionIndexer` (CPU)
*   **Inputs:** `EnrichedSRT`, `Scene` objects, `VisualText`.
*   **Logic:**
    1.  **Time Alignment:** Map SRT lines to Scenes based on timestamp overlap.
    2.  **Aggregation:** Join all dialogue in scene -> `dialogue_text`. Join all captions -> `visual_description`.
    3.  **Embedding:** Run `all-MiniLM-L6-v2` locally (CPU or minimal GPU) on both text fields.
    4.  **Upsert:** Send to Qdrant.

---

## 3. Directory Structure Updates

We will add these specific files to implement the spec:

```text
src/
├── components/
│   ├── metadata/
│   │   ├── __init__.py
│   │   ├── tmdb_client.py       # Task 1
│   │   └── radarr_client.py     # Task 1
│   ├── vision/                  # NEW COMPONENT
│   │   ├── scene_detector.py    # Task 2 (PySceneDetect)
│   │   ├── face_identifier.py   # Task 3 (InsightFace)
│   │   └── visual_describer.py  # Task 3 (Florence-2)
```

---

# 🥊 LLM 2 Review & "Sparring"

**Role:** Senior Architect / Critic
**Review Status:** **APPROVED** (with 2 caveats)

### Critique 1: The "Florence Context" Assumption
In section 2.4, you suggest "If FaceID says Ian McKellen... append to caption."
*   **Risk:** Florence might describe "An old man in white." If you blindly append "featuring Ian McKellen," the search "Ian McKellen in white" works. BUT, if Florence fails to describe the person at all (e.g., "A wide shot of a mountain"), appending "featuring Ian McKellen" creates a **Hallucination** in the index. The user searches for Ian, gets a mountain.
*   **Fix:** Only append actor names if a person/face is actually detected in that specific frame. (The `FaceIdentifier` output already guarantees this, so just ensure strict frame-level mapping, not scene-level smearing).

### Critique 2: Plex `rating_key` Retrieval
*   **Risk:** Radarr API does not always provide the Plex `ratingKey` unless they are explicitly linked via a notification agent.
*   **Fix:** The `MetadataService` needs a direct fallback mechanism to query the **Plex API** directly using the filename if Radarr fails to provide the ID.
*   **Refinement:** Add `plex_client.py` to `src/components/metadata/`.

### Consensus
**LLM 1 (Me):** I accept both critiques.
1.  I will ensure the Fusion logic only tags faces to the visual description if the face was detected in the specific shots used for description.
2.  I will add `plex_client.py` to the Task 1 scope to ensure we can generate valid deep links.

---

## ✅ Final Confirmation

We have:
1.  **High Level Strategy:** "The Portable Director" (Laptop Optimized).
2.  **Detailed Specs:** 2-Pass GPU pipeline, specific models (Florence-2/InsightFace), clear Qdrant schema.
3.  **Agreement:** Both personas agree this is ready to code.

**Are you ready to begin with Task 1: The Metadata Service (Radarr/TMDB/Plex Clients)?**

---

# 🔬 LLM 2: Detailed Implementation Review

## Code Integration Analysis

I've reviewed the existing Memoirr codebase. Here's how the V1 spec maps to what already exists, what needs modification, and potential conflicts.

---

## 1. Qdrant Schema: Breaking Change Alert 🚨

### Current State
The existing `QdrantWriter` in `src/components/writer/qdrant_writer.py` creates a **single-vector collection**:
- Uses Haystack's `QdrantDocumentStore` which assumes one embedding per document
- Vector dimension comes from `EMBEDDING_DIMENSION` setting (currently 1024 for Qwen3)
- Payload follows Haystack's `Document` structure: `{content, meta, embedding}`

### V1 Spec Requires
- **Named vectors**: `text` (384d) + `visual` (384d)
- **Custom payload**: `speakers`, `faces_present`, `visual_keywords`, etc.
- **Different embedding model**: `all-MiniLM-L6-v2` (384d) vs current Qwen3 (1024d)

### 🔴 CONFLICT: Model Dimension Mismatch
The spec says use `all-MiniLM-L6-v2` (384d) for both text and visual vectors. But:
- Current system uses Qwen3-embedding-0.6B (1024d)
- Qwen3 produces significantly better semantic search quality
- Switching to MiniLM is a **downgrade** for text search

### My Recommendation
**Keep Qwen3 for text, use MiniLM only for visual descriptions (if VRAM constrained):**
```python
# Proposed vector config
"text": 1024 dimensions (Qwen3-embedding-0.6B) - high quality text search
"visual": 384 dimensions (all-MiniLM-L6-v2) - lightweight visual desc embedding
```

Or better: **Use the same Qwen3 model for both** if VRAM allows. The 0.6B model is only ~200MB VRAM. Consistency matters more than saving 100MB.

**Action Required:** Confirm embedding model strategy before implementation.

---

## 2. Component Architecture: Fits Well ✅

The proposed structure aligns with existing patterns:

| Proposed Component | Matches Pattern | Location |
|-------------------|-----------------|----------|
| `MetadataService` | ✅ New component | `src/components/metadata/` |
| `SceneSegmenter` | ✅ New component | `src/components/vision/` |
| `FaceIdentifier` | ✅ New component | `src/components/vision/` |
| `VisualDescriber` | ✅ New component | `src/components/vision/` |
| `SpeakerTagger` | ⚠️ Extends `GroqGenerator` | `src/components/generator/` |
| `FusionIndexer` | ⚠️ Replaces `QdrantWriter` | `src/components/writer/` |

### Concern: SpeakerTagger vs GroqGenerator
The existing `GroqGenerator` in `src/components/generator/groq_generator.py` is designed for RAG response generation, not structured JSON extraction. The `SpeakerTagger` needs:
- Different prompt template (JSON output schema)
- Different response parsing (extract speaker attributions)
- Sliding window orchestration

**Recommendation:** Create `SpeakerTagger` as a **separate component**, not a subclass. It shares the Groq API but has fundamentally different I/O contracts.

---

## 3. Configuration Integration ✅

New settings needed in `src/core/config.py`:

```python
# Metadata API settings
tmdb_api_key: Optional[str] = Field(default=None, alias="TMDB_API_KEY")
radarr_url: Optional[str] = Field(default=None, alias="RADARR_URL")
radarr_api_key: Optional[str] = Field(default=None, alias="RADARR_API_KEY")
plex_url: Optional[str] = Field(default=None, alias="PLEX_URL")
plex_token: Optional[str] = Field(default=None, alias="PLEX_TOKEN")

# Vision pipeline settings
scene_detect_threshold: float = Field(default=27.0, alias="SCENE_DETECT_THRESHOLD")
scene_min_duration_sec: float = Field(default=1.5, alias="SCENE_MIN_DURATION_SEC")
keyframes_per_scene: int = Field(default=3, alias="KEYFRAMES_PER_SCENE")
face_similarity_threshold: float = Field(default=0.6, alias="FACE_SIMILARITY_THRESHOLD")

# Model paths (for InsightFace, Florence)
insightface_model: str = Field(default="buffalo_l", alias="INSIGHTFACE_MODEL")
florence_model: str = Field(default="microsoft/Florence-2-large", alias="FLORENCE_MODEL")

# Speaker tagging settings
speaker_confidence_threshold: float = Field(default=0.8, alias="SPEAKER_CONFIDENCE_THRESHOLD")
speaker_sliding_window: int = Field(default=20, alias="SPEAKER_SLIDING_WINDOW")
```

This follows the existing pattern. No conflicts.

---

## 4. Pipeline Integration: New Pipeline Needed

### Current Pipeline
`src/pipelines/srt_to_qdrant.py` does:
```
SRT → SrtPreprocessor → SemanticChunker → TextEmbedder → QdrantWriter
```

### V1 Pipeline
Need a new `video_to_qdrant.py`:
```
Video + SRT → MetadataService ─┬─→ SceneSegmenter → FaceIdentifier → VisualDescriber ─┐
                               │                                                        │
                               └─→ SrtPreprocessor → SpeakerTagger ────────────────────┤
                                                                                        │
                                                              FusionIndexer ←───────────┘
                                                                    │
                                                                 Qdrant
```

This is a **DAG pipeline**, not linear. Haystack 2.x supports this via `Pipeline.connect()` with branching.

### Code Skeleton
```python
from haystack import Pipeline

pipeline = Pipeline()
pipeline.add_component("metadata", MetadataService())
pipeline.add_component("scene_seg", SceneSegmenter())
pipeline.add_component("face_id", FaceIdentifier())
pipeline.add_component("visual_desc", VisualDescriber())
pipeline.add_component("srt_prep", SrtPreprocessor())
pipeline.add_component("speaker_tag", SpeakerTagger())
pipeline.add_component("fusion", FusionIndexer())

# Parallel branches
pipeline.connect("metadata.cast_list", "face_id.cast_list")
pipeline.connect("metadata.cast_list", "speaker_tag.cast_list")
pipeline.connect("scene_seg.scenes", "face_id.scenes")
pipeline.connect("face_id.face_mapping", "visual_desc.face_mapping")
pipeline.connect("scene_seg.frames", "visual_desc.frames")
pipeline.connect("srt_prep.jsonl_lines", "speaker_tag.dialogue")

# Fusion inputs
pipeline.connect("visual_desc.descriptions", "fusion.visual")
pipeline.connect("speaker_tag.enriched_srt", "fusion.dialogue")
pipeline.connect("scene_seg.scenes", "fusion.scenes")
pipeline.connect("metadata.movie_meta", "fusion.metadata")
```

---

## 5. VRAM Management: Critical Missing Piece 🚨

The spec mentions `torch.cuda.empty_cache()` but the current codebase has better tooling in `src/core/memory_utils.py`:

```python
from src.core.memory_utils import clear_gpu_memory, log_memory_usage, check_memory_availability
```

### Required Pattern for 2-Pass GPU Pipeline
```python
# Pass 1: Face Identification
face_identifier = FaceIdentifier()
face_mapping = face_identifier.run(frames, cast_list)
del face_identifier  # Release model
clear_gpu_memory()   # Force VRAM release
log_memory_usage("after face identification", logger)

# Pass 2: Visual Description
visual_describer = VisualDescriber()
descriptions = visual_describer.run(frames, face_mapping)
del visual_describer
clear_gpu_memory()
```

**Action Required:** Each GPU component MUST:
1. Accept a `cleanup: bool = True` parameter
2. Call `clear_gpu_memory()` in `__del__` or explicit cleanup method
3. Log VRAM usage before/after

---

## 6. Missing Utility Functions

The spec assumes utilities that don't exist yet:

| Utility | Purpose | Suggested Location |
|---------|---------|-------------------|
| `parse_filename(path) -> MovieGuess` | Extract title/year from filename | `src/components/metadata/utilities/filename_parser.py` |
| `align_srt_to_scenes(srt, scenes) -> Mapping` | Time-based alignment | `src/components/writer/utilities/time_alignment.py` |
| `cluster_faces(embeddings) -> Clusters` | DBSCAN clustering | `src/components/vision/utilities/face_clustering.py` |
| `match_faces_to_actors(clusters, headshots) -> Mapping` | Cosine similarity matching | `src/components/vision/utilities/face_matching.py` |

These follow the 3-layer pattern (Component → Orchestrator → Utilities).

---

## 7. Type Definitions Needed

Create `src/components/vision/utilities/types.py`:

```python
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path

@dataclass(frozen=True)
class Scene:
    """A detected scene from video."""
    scene_id: int
    start_ms: int
    end_ms: int
    keyframe_paths: List[Path]

@dataclass(frozen=True)
class FaceDetection:
    """A detected face in a frame."""
    frame_path: Path
    bbox: tuple  # (x1, y1, x2, y2)
    embedding: List[float]  # 512d ArcFace
    confidence: float

@dataclass(frozen=True)
class ActorMatch:
    """A face cluster matched to an actor."""
    cluster_id: int
    actor_name: str
    character_name: str
    confidence: float
    scene_ids: List[int]

@dataclass(frozen=True)
class MovieMetadata:
    """Enriched movie metadata from TMDB/Radarr."""
    title: str
    year: int
    tmdb_id: int
    plex_rating_key: Optional[str]
    cast: List[Dict[str, str]]  # [{actor, character, profile_url}]
    genres: List[str]
```

---

## 8. Testing Strategy

Following existing patterns in `test/components/`:

| Component | Test File | Key Test Cases |
|-----------|-----------|----------------|
| `tmdb_client` | `test_tmdb_client.py` | Mock API responses, rate limiting, missing movies |
| `radarr_client` | `test_radarr_client.py` | Mock API, filename matching |
| `plex_client` | `test_plex_client.py` | Deep link generation, auth |
| `scene_detector` | `test_scene_detector.py` | Mock PySceneDetect, edge cases (short video, no scenes) |
| `face_identifier` | `test_face_identifier.py` | Mock InsightFace, clustering validation |
| `visual_describer` | `test_visual_describer.py` | Mock Florence, caption quality |
| `speaker_tagger` | `test_speaker_tagger.py` | Mock Groq, confidence tiers, sliding window |
| `fusion_indexer` | `test_fusion_indexer.py` | Time alignment, Qdrant upsert with named vectors |

**Critical:** All GPU components need `@pytest.mark.skipif(not torch.cuda.is_available())` for CI.

---

## 9. Dependencies to Add

Update `pyproject.toml`:

```toml
[project.dependencies]
# Existing...

# New for V1
scenedetect = "^0.6"          # PySceneDetect
insightface = "^0.7"          # Face recognition
onnxruntime-gpu = "^1.16"     # InsightFace backend (GPU)
opencv-python = "^4.8"        # Frame extraction
requests = "^2.31"            # API clients
plexapi = "^4.15"             # Plex integration

[project.optional-dependencies]
vision = [
    "transformers>=4.36",     # Florence-2
    "accelerate>=0.25",       # Model loading
]
```

---

## 10. Implementation Order (Refined)

Based on code review, here's the dependency-aware order:

### Phase 1: Foundation (No GPU)
1. **Config updates** - Add new settings to `config.py`
2. **Types** - Create `vision/utilities/types.py`
3. **TMDB Client** - Pure API, easy to test
4. **Radarr Client** - Pure API
5. **Plex Client** - Pure API + deep link generation
6. **Filename Parser** - Utility for fallback

### Phase 2: Scene Pipeline (GPU Pass 1 prep)
7. **SceneSegmenter** - PySceneDetect wrapper, extracts keyframes
8. **Time Alignment Utility** - Map SRT to scenes

### Phase 3: GPU Components
9. **FaceIdentifier** - InsightFace + DBSCAN clustering
10. **VisualDescriber** - Florence-2 captioning
11. **Integration test** - Verify VRAM fits on 4060

### Phase 4: Speaker & Fusion
12. **SpeakerTagger** - Groq-based, reuse existing patterns
13. **FusionIndexer** - New QdrantWriter with named vectors

### Phase 5: Pipeline & UI
14. **video_to_qdrant.py** - Full DAG pipeline
15. **Hybrid search** - Update `QdrantRetriever` for named vectors
16. **Gradio updates** - Add visual search UI

---

## ❓ Questions Before Implementation

1. **Embedding model decision:** Keep Qwen3 (1024d) for text, or switch to MiniLM (384d) for consistency?

2. **Existing data migration:** Will we have two collections (`memoirr` for text-only, `memoirr_v1` for multimodal) or migrate?

3. **Florence model size:** The spec says `Florence-2-large` (2.5GB VRAM). On 8GB with InsightFace, this is tight. Should we use `Florence-2-base` (0.7GB) instead for safety margin?

4. **InsightFace model:** Spec says `buffalo_l` (640px, ~1.2GB). Should we use `buffalo_s` (~400MB) for 4060?

5. **Temp file cleanup:** Keyframes go to `/tmp/memoirr/`. Add cleanup in pipeline or keep for debugging?

---

## ✅ Ready to Implement

Once the above questions are answered, Phase 1 can begin immediately. The foundation (API clients, types, config) has zero GPU dependencies and can be fully tested.

**Recommended first PR:** 
- `src/components/metadata/` (tmdb_client, radarr_client, plex_client)
- `src/core/config.py` updates
- Tests with mocked APIs

This unblocks all downstream work.

---

*Awaiting answers to proceed with implementation.*

---

# 📋 Implementation Steps: High-Level Breakdown

The following steps are designed to be handed to LLM 1 for detailed specification. Each step should result in a document with:
- Exact file paths to create/modify
- Function signatures and types
- Test cases to write
- Dependencies and prerequisites

---

## Phase 1: Foundation (No GPU Required)

### Step 1.1: Configuration Updates
**Goal:** Add all new settings to the config system.

**Scope:**
- Update `src/core/config.py` with TMDB, Radarr, Plex, and vision pipeline settings
- Add validation for API keys (warn if missing, don't crash)
- Update `.env.example` with new variables

**Output:** Config ready to support all new components

---

### Step 1.2: Vision Type Definitions
**Goal:** Create frozen dataclasses for all vision pipeline data structures.

**Scope:**
- Create `src/components/vision/utilities/types.py`
- Define: `Scene`, `FaceDetection`, `FaceCluster`, `ActorMatch`, `VisualDescription`
- Create `src/components/metadata/utilities/types.py`
- Define: `MovieMetadata`, `CastMember`, `PlexDeepLink`

**Output:** Type-safe data contracts for all new components

---

### Step 1.3: TMDB Client
**Goal:** Fetch movie metadata and cast information from TMDB API.

**Scope:**
- Create `src/components/metadata/tmdb_client.py`
- Functions: `search_movie(title, year)`, `get_movie_details(tmdb_id)`, `get_cast(tmdb_id)`, `get_actor_headshot_url(person_id)`
- Handle rate limiting, missing movies, API errors
- Create orchestrator and utility layer per coding standards

**Output:** Can fetch metadata for any movie by title/year

---

### Step 1.4: Radarr Client
**Goal:** Look up movies in user's Radarr library to get file paths and existing metadata.

**Scope:**
- Create `src/components/metadata/radarr_client.py`
- Functions: `get_movie_by_tmdb_id(tmdb_id)`, `get_all_movies()`, `get_movie_file_path(radarr_id)`
- Handle connection errors, missing library

**Output:** Can map TMDB movies to local file paths

---

### Step 1.5: Plex Client
**Goal:** Generate deep links for playback and optionally fetch Plex metadata.

**Scope:**
- Create `src/components/metadata/plex_client.py`
- Functions: `search_movie(title, year)`, `get_rating_key(title, year)`, `generate_deep_link(rating_key, timestamp_ms)`
- Support both web URLs and plex:// protocol

**Output:** Can generate playable links to specific timestamps

---

### Step 1.6: Filename Parser Utility
**Goal:** Extract movie title and year from filename when APIs fail.

**Scope:**
- Create `src/components/metadata/utilities/filename_parser.py`
- Use `guessit` library or regex patterns
- Handle common naming conventions: `Movie.Name.2020.1080p.BluRay.x264.mkv`

**Output:** Fallback metadata extraction for files not in TMDB

---

### Step 1.7: Metadata Service Component
**Goal:** Haystack component that orchestrates all metadata sources.

**Scope:**
- Create `src/components/metadata/metadata_service.py`
- Follows 3-layer pattern with orchestrator
- Input: video file path or movie title/year
- Output: `MovieMetadata` with cast list, genres, Plex rating key
- Priority: Radarr → TMDB → filename parsing

**Output:** Single component that provides complete movie metadata

---

## Phase 2: Scene Pipeline (No GPU Required)

### Step 2.1: Scene Segmentation Component
**Goal:** Detect scene boundaries and extract keyframes from video files.

**Scope:**
- Create `src/components/vision/scene_segmenter.py`
- Wrap PySceneDetect library
- Input: video file path
- Output: List of `Scene` objects with keyframe paths
- Config: threshold, min scene duration, keyframes per scene
- Save keyframes to temp directory

**Output:** Can split any video into scenes with representative frames

---

### Step 2.2: Time Alignment Utility
**Goal:** Map SRT dialogue entries to detected scenes.

**Scope:**
- Create `src/components/vision/utilities/time_alignment.py`
- Function: `align_srt_to_scenes(srt_entries, scenes) -> Dict[scene_id, List[SrtEntry]]`
- Handle overlapping dialogue, dialogue spanning scene boundaries
- Return scene_id for each dialogue entry

**Output:** Every dialogue line knows which scene it belongs to

---

## Phase 3: GPU Components (RTX 4060 Compatible)

### Step 3.1: Face Identifier Component
**Goal:** Detect faces in keyframes and cluster them by identity.

**Scope:**
- Create `src/components/vision/face_identifier.py`
- Use InsightFace with ArcFace model
- Input: keyframe paths, cast list (for labeling)
- Output: `Dict[cluster_id, ActorMatch]` mapping face clusters to characters
- Implement VRAM management (load/unload model)
- Match clusters to TMDB headshots via cosine similarity

**Output:** Can identify which characters appear in which scenes

---

### Step 3.2: Visual Describer Component
**Goal:** Generate text descriptions of keyframes using VLM.

**Scope:**
- Create `src/components/vision/visual_describer.py`
- Use Florence-2-base model
- Input: keyframe paths, face mapping (to inject character names)
- Output: List of `VisualDescription` per scene
- Prompt engineering: include character names in context
- Implement VRAM management

**Output:** Text descriptions of what's happening visually in each scene

---

### Step 3.3: GPU Memory Integration Test
**Goal:** Verify both GPU components fit on RTX 4060 (8GB).

**Scope:**
- Create integration test that runs both components sequentially
- Log VRAM at each stage
- Verify cleanup works between components
- Test with real video file (short clip)

**Output:** Confidence that pipeline runs on target hardware

---

## Phase 4: Speaker Attribution & Fusion

### Step 4.1: Speaker Tagger Component
**Goal:** Attribute dialogue lines to characters using LLM.

**Scope:**
- Create `src/components/generator/speaker_tagger.py`
- Use Groq API (existing pattern)
- Input: SRT entries, cast list, scene context
- Output: Enriched SRT with `speaker` and `confidence` fields
- Implement sliding window for context
- Three confidence tiers: confirmed, high_confidence, inferred

**Output:** Every dialogue line has a speaker attribution

---

### Step 4.2: Fusion Indexer Component
**Goal:** Combine all enriched data and write to Qdrant with named vectors.

**Scope:**
- Create `src/components/writer/fusion_indexer.py`
- Input: scenes, visual descriptions, enriched SRT, metadata
- Create Qdrant collection with named vectors (`text`, `visual`)
- Build payload with all searchable fields
- Embed text with Qwen3, visual descriptions with same or MiniLM
- Upsert to Qdrant

**Output:** Fully indexed movie in Qdrant with multimodal search capability

---

### Step 4.3: Updated Qdrant Schema
**Goal:** Define and document the new collection schema.

**Scope:**
- Document named vector configuration
- Document payload fields and their purposes
- Create schema migration utility (if upgrading existing collections)
- Add index configuration for filterable fields

**Output:** Clear schema documentation and creation code

---

## Phase 5: Search & Pipeline Integration

### Step 5.1: Hybrid Retriever Updates
**Goal:** Update retriever to search both text and visual vectors.

**Scope:**
- Update `src/components/retriever/qdrant_retriever.py`
- Add `search_mode` parameter: `text`, `visual`, `hybrid`
- Implement RRF fusion for hybrid mode
- Return unified ranked results

**Output:** Single retriever that searches across modalities

---

### Step 5.2: Video Indexing Pipeline
**Goal:** Create end-to-end pipeline from video file to indexed Qdrant.

**Scope:**
- Create `src/pipelines/video_to_qdrant.py`
- Wire all components in correct order
- Handle partial failures gracefully
- Add progress logging
- Support batch processing of multiple videos

**Output:** One command to index a movie

---

### Step 5.3: Updated RAG Pipeline
**Goal:** Integrate multimodal search into existing RAG flow.

**Scope:**
- Update `src/pipelines/rag_pipeline.py`
- Add support for visual queries
- Include Plex deep links in responses
- Format responses with speaker attributions and confidence

**Output:** Chat interface that leverages all new capabilities

---

### Step 5.4: Gradio UI Updates
**Goal:** Add visual search UI and display enhancements.

**Scope:**
- Update `src/frontend/gradio_app.py`
- Add search mode toggle (text/visual/hybrid)
- Display scene thumbnails in results
- Show speaker attributions with confidence indicators
- Add "Play in Plex" buttons with deep links

**Output:** User-facing interface for all V1 features

---

## Summary: 18 Implementation Steps

| Phase | Steps | GPU Required | Estimated Complexity |
|-------|-------|--------------|---------------------|
| Phase 1: Foundation | 1.1 - 1.7 (7 steps) | No | Low-Medium |
| Phase 2: Scene Pipeline | 2.1 - 2.2 (2 steps) | No | Medium |
| Phase 3: GPU Components | 3.1 - 3.3 (3 steps) | Yes | High |
| Phase 4: Speaker & Fusion | 4.1 - 4.3 (3 steps) | No (API) | Medium-High |
| Phase 5: Integration | 5.1 - 5.4 (4 steps) | No | Medium |

---

**Next Action:** LLM 1 should pick a step (recommend starting with 1.1) and write detailed implementation instructions including exact code signatures, test cases, and acceptance criteria.

---

*Ready for LLM 1 to begin detailed specifications.*