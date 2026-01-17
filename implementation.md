# Memoirr V1 Multimodal Roadmap

**Goal:** Transform Memoirr from a text-based subtitle search into a multimodal media RAG system capable of visual search ("Find the scene with the white horse") and deep playback linking.

**Hardware Target:** Single Node (RTX 4060 Laptop, 8GB VRAM).
**Architecture:** Sequential Pipeline (Metadata -> Vision -> Text -> Fusion).

---
## 1. Model Bill of Materials (BOM)

These are the specific models we will use. They are selected to fit within 8GB VRAM when run one at a time.

| Role | Selected Model | Implementation | Resource Est. |
| :--- | :--- | :--- | :--- |
| **Visual Describer** | `Qwen/Qwen2.5-VL-3B-Instruct` | `transformers` + `bitsandbytes` (4-bit) | ~3.5 GB VRAM |
| **Face Recognition** | `InsightFace (buffalo_l)` | `onnxruntime-gpu` | ~1.2 GB VRAM |
| **Text Embedding** | `Qwen/Qwen3-embedding-0.6B` | `sentence-transformers` | ~0.6 GB VRAM |
| **Speaker Tagging** | `Llama-3-8b` | **Groq API** (Cloud) | 0 GB VRAM |

---

## 📅 Phase 1: Foundation (Infrastructure & Metadata)
**Focus:** Configuration, Type Safety, and External APIs. No GPU required.

- [ ] **Step 1.1: Configuration Updates**
    - **Goal:** Centralize new API keys and model settings.
    - **File:** `src/core/config.py`
    - **Scope:** Add TMDB, Radarr, Plex credentials; Add Vision pipeline thresholds; Update `.env.example`.

- [ ] **Step 1.2: Vision & Metadata Type Definitions**
    - **Goal:** Create frozen dataclasses for data contracts.
    - **Files:** `src/components/vision/utilities/types.py`, `src/components/metadata/utilities/types.py`
    - **Scope:** Define `Scene`, `FaceDetection`, `MovieMetadata`, `PlexDeepLink`.

- [ ] **Step 1.3: TMDB Client**
    - **Goal:** Fetch cast lists and headshots.
    - **File:** `src/components/metadata/tmdb_client.py`
    - **Scope:** API client for Movie Details, Credits, and Images.

- [ ] **Step 1.4: Radarr Client**
    - **Goal:** Map files on disk to metadata IDs.
    - **File:** `src/components/metadata/radarr_client.py`
    - **Scope:** Lookup movie by path; Get TMDB/Plex IDs.

- [ ] **Step 1.5: Plex Client**
    - **Goal:** Enable "Click to Play".
    - **File:** `src/components/metadata/plex_client.py`
    - **Scope:** Generate `plex://` deep links for specific timestamps.

- [ ] **Step 1.6: Metadata Service Component**
    - **Goal:** Haystack component wrapping the above clients.
    - **File:** `src/components/metadata/metadata_service.py`
    - **Scope:** Orchestrate fetching metadata -> outputs `MovieMetadata` object.

---

## 🎬 Phase 2: Scene Pipeline (CPU Processing)
**Focus:** Video manipulation and temporal alignment.

- [ ] **Step 2.1: Scene Segmentation Component**
    - **Goal:** Break video into logical shots and extract keyframes.
    - **File:** `src/components/vision/scene_segmenter.py`
    - **Dependencies:** `PySceneDetect`, `ffmpeg`
    - **Scope:** Adaptive threshold detection; Keyframe extraction (and resizing to 720p).

- [ ] **Step 2.2: Time Alignment Utility**
    - **Goal:** Map subtitle lines to visual scenes.
    - **File:** `src/components/vision/utilities/time_alignment.py`
    - **Scope:** Algorithm to assign dialogue lines to specific Scene IDs based on timestamp overlap.

---

## 🧠 Phase 3: GPU Components (The Heavy Lift)
**Focus:** Visual understanding using local AI models. Strictly sequential execution to fit 8GB VRAM.

- [ ] **Step 3.1: Face Identifier Component**
    - **Goal:** Know *who* is in the frame.
    - **File:** `src/components/vision/face_identifier.py`
    - **Model:** InsightFace (`buffalo_l`)
    - **Scope:** Detect faces -> Cluster embeddings -> Match clusters to TMDB Actor Headshots.

- [ ] **Step 3.2: Visual Describer Component**
    - **Goal:** Know *what* is happening.
    - **File:** `src/components/vision/visual_describer.py`
    - **Model:** `Florence-2-large` (or `base` if VRAM tight)
    - **Scope:** Dense Captioning prompt; Context injection ("Describe this scene featuring [Actor Name]...").

- [ ] **Step 3.3: GPU Memory Integration Test**
    - **Goal:** Verify the laptop doesn't crash.
    - **File:** `test/integration/test_gpu_pipeline_memory.py`
    - **Scope:** Run 3.1 followed by 3.2 on a dummy video; Assert VRAM cleanup between stages.

---

## 🗣️ Phase 4: Semantic Enrichment & Fusion
**Focus:** Intelligent text processing and database indexing.

- [ ] **Step 4.1: Speaker Tagger Component**
    - **Goal:** Attribute dialogue to characters.
    - **File:** `src/components/generator/speaker_tagger.py`
    - **Model:** `Llama-3` (via Groq API)
    - **Scope:** Sliding window prompt ("Assign these lines to these actors..."); Confidence scoring.

- [ ] **Step 4.2: Fusion Indexer Component**
    - **Goal:** Construct the "Scene Node" and save to DB.
    - **File:** `src/components/writer/fusion_indexer.py`
    - **Scope:** Aggregate Data (Metadata + Scenes + Faces + Dialogue); Embed Text (Qwen3); Embed Visual Description (Qwen3); Upsert to Qdrant (Named Vectors).

- [ ] **Step 4.3: Qdrant Schema Definition**
    - **Goal:** Define the new collection structure.
    - **Document:** `docs/specs/qdrant_schema_v1.md`
    - **Scope:** Define named vectors config and payload index requirements.

---

## 🚀 Phase 5: Integration & UI
**Focus:** Wiring it together and delivering user value.

- [ ] **Step 5.1: Video Indexing Pipeline**
    - **Goal:** The "One Command" ingest script.
    - **File:** `src/pipelines/video_to_qdrant.py`
    - **Scope:** Connect all components (Phases 1-4) into a linear Haystack DAG.

- [ ] **Step 5.2: Hybrid Retriever Update**
    - **Goal:** Search Text + Visuals simultaneously.
    - **File:** `src/components/retriever/qdrant_retriever.py`
    - **Scope:** Add Reciprocal Rank Fusion (RRF) logic to merge visual/text hits.

- [ ] **Step 5.3: Gradio UI Upgrade**
    - **Goal:** Show off the results.
    - **File:** `src/frontend/gradio_app.py`
    - **Scope:** Display "Scene Card" (Thumbnail, Who is present, Dialogue, "Play" button).