"""Global configuration using Pydantic Settings.

Loads environment variables (including from a .env file) and provides
centralized access for components and utilities.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings.

    Attributes:
        embedding_model_name: Local model folder name under ./models where
            model.safetensors and tokenizer files live.
        device: Optional device string for transformers (e.g., "cuda:0", "cpu").

        Chunker configuration (can be overridden via .env):
        - CHUNK_THRESHOLD: float/int/"auto" (string); default "auto".
        - CHUNK_SIZE: int; default 512.
        - CHUNK_SIMILARITY_WINDOW: int; default 3.
        - CHUNK_MIN_SENTENCES: int; default 2.
        - CHUNK_MIN_CHARACTERS_PER_SENTENCE: int; default 24.
        - CHUNK_DELIM: JSON array string or plain string; default JSON list of common delimiters.
        - CHUNK_INCLUDE_DELIM: "prev" | "next" | "none"; default "prev".
        - CHUNK_SKIP_WINDOW: int; default 0.
        - CHUNK_INCLUDE_PARAMS: bool; default True.
        - CHUNK_INCLUDE_CAPTION_INDICES: bool; default True.
        - CHUNK_FAIL_FAST: bool; default True.
    """

    embedding_model_name: str = Field(
        default="qwen3-embedding-0.6B", alias="EMBEDDING_MODEL_NAME"
    )
    device: str | None = Field(default=None, alias="EMBEDDING_DEVICE")
    embedding_dimension: int | None = Field(default=None, alias="EMBEDDING_DIMENSION")

    # Chunker settings
    chunk_threshold: str = Field(default="auto", alias="CHUNK_THRESHOLD")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_similarity_window: int = Field(default=3, alias="CHUNK_SIMILARITY_WINDOW")
    chunk_min_sentences: int = Field(default=2, alias="CHUNK_MIN_SENTENCES")
    chunk_min_characters_per_sentence: int = Field(default=24, alias="CHUNK_MIN_CHARACTERS_PER_SENTENCE")
    chunk_delim: str = Field(default='[". ", "! ", "? ", "\\n"]', alias="CHUNK_DELIM")
    chunk_include_delim: str | None = Field(default="prev", alias="CHUNK_INCLUDE_DELIM")
    chunk_skip_window: int = Field(default=0, alias="CHUNK_SKIP_WINDOW")
    chunk_include_params: bool = Field(default=True, alias="CHUNK_INCLUDE_PARAMS")
    chunk_include_caption_indices: bool = Field(default=True, alias="CHUNK_INCLUDE_CAPTION_INDICES")
    chunk_fail_fast: bool = Field(default=True, alias="CHUNK_FAIL_FAST")

    # Preprocessor settings
    pre_min_len: int = Field(default=1, alias="PRE_MIN_LEN")
    pre_dedupe_window_ms: int = Field(default=1000, alias="PRE_DEDUPE_WINDOW_MS")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()  # type: ignore[call-arg]
